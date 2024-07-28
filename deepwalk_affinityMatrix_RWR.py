import numpy as np
import networkx as nx
import random
from tqdm import tqdm
import torch
from scipy.sparse import csr_matrix, diags
from gensim.models import Word2Vec

class PersonalizedRW:
    # 初始化函数，输入节点数目和边的txt文件地址，以及是否为无向图
    def __init__(self, node_num: int, edge_txt: str, undirected=False, device='cpu'):
        self.device = torch.device(device)
        if torch.cuda.is_available() and device == 'cuda':
            print("使用GPU进行计算")
        else:
            print("使用CPU进行计算")

        print("初始化图结构")
        if undirected:
            self.G = nx.Graph()  # 无向图
        else:
            self.G = nx.DiGraph()  # 有向图
        self.G.add_nodes_from(list(range(node_num)))  # 添加节点
        edges = self.read_edge(edge_txt)  # 读取边信息
        self.G.add_edges_from(edges)  # 添加边
        self.adjacency = csr_matrix(nx.adjacency_matrix(self.G))  # 计算邻接矩阵并转换为压缩稀疏行矩阵（CSR）
        self.transition_matrix = self.get_transition_matrix()  # 计算概率转移矩阵

    # 读取边信息的函数
    def read_edge(self, edge_txt):
        print("读取边信息")
        edges = np.loadtxt(edge_txt, dtype=np.int16)  # 读取txt文件中的边信息
        edges = [(edges[i, 0], edges[i, 1]) for i in range(edges.shape[0])]  # 将每条边转换为元组
        return edges

    # 计算概率转移矩阵的函数
    def get_transition_matrix(self):
        print("计算概率转移矩阵")
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()  # 计算每个节点的度
        degrees[degrees == 0] = 1  # 将度为0的节点度设置为1，避免除以0，后面不影响计算0/1结果还是0
        D_inv = diags(1.0 / degrees)  # 构建度的逆对角矩阵
        P = D_inv.dot(self.adjacency)  # 计算概率转移矩阵 P = D^{-1} * A
        return torch.tensor(P.toarray(), dtype=torch.float32).to(self.device)  # 转换为PyTorch张量并移动到GPU

    # 计算个性化随机游走的亲和矩阵并归一化
    # 参考公式：https://lovexl-oss.oss-cn-beijing.aliyuncs.com/bed/202407251421774.png
    def personalized_random_walk(self, alpha=0, max_iter=1000):
        print("     计算亲和矩阵，max_iter为", max_iter)
        P = self.transition_matrix
        M = torch.zeros_like(P).to(self.device)
        power_P = torch.eye(P.shape[0], device=self.device)  # 创建一个单位矩阵，P^0 = I
        
        if alpha == 0:
            for k in tqdm(range(max_iter), desc="Calculating affinity matrix (alpha=0)"):
                M += power_P  # 当 alpha = 0 时，M 为所有 P^k 的累加
                power_P = power_P.matmul(P)  # 计算 P^k
        else:
            for k in tqdm(range(max_iter), desc="Calculating affinity matrix"):
                M += alpha * (1 - alpha) ** k * power_P  # 计算亲和矩阵 M
                power_P = power_P.matmul(P)  # 计算 P^k

        # 对亲和矩阵进行归一化
        row_sums = M.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        M = M / row_sums
        
        return M

    # 构建语料库的函数
    def build_corpus(self, path_len, alpha=0, max_iter=1000, rand_iter=random.Random(628)):
        print("开始生成个性化随机游走语料库：")
        total_walks = []
        M = self.personalized_random_walk(alpha, max_iter)  # 计算亲和矩阵，不移动到CPU
        M_cpu = M.cpu().numpy()  # 仅一次性转换整个矩阵到CPU
        nodes = list(self.G.nodes)  # 获取所有节点列表
        print("     开始生成随机游走序列")
        for _ in tqdm(range(50), desc="Generating random walks"):
            for start_node in nodes:
                walk = self.random_walk_from_matrix(M_cpu, start_node, path_len, rand_iter)  # 从亲和矩阵中生成随机游走序列
                total_walks.append(walk)
                
        # 打印亲和矩阵的前5行
        print("     亲和矩阵的前5行为：", M[:5])
        # 打印生成的随机游走序列的前5个
        print("     生成的随机游走序列的前5个为：", total_walks[:5])
        return total_walks

    # 从亲和矩阵中生成随机游走序列
    def random_walk_from_matrix(self, M_cpu, start, path_len, rand_iter):
        walk = [start]
        current_node = start
        for _ in range(path_len - 1):
            weights = M_cpu[current_node]
            weights_sum = weights.sum()
            if weights_sum == 0:
                # 如果权重总和为0，说明没有有效的下一个节点，停止游走
                break
            next_node = rand_iter.choices(range(M_cpu.shape[0]), weights=weights, k=1)[0]  # 根据归一化后的权重选择下一个节点
            walk.append(next_node)
            current_node = next_node
        return [str(node) for node in walk]

    # 训练Word2Vec模型
    def train(self, total_walks, embed_size=64, window_size=3, output="."):
        print("开始训练word2vec模型")
        model = Word2Vec(
            total_walks,
            vector_size=embed_size,  # 词向量维度
            window=window_size,  # 窗口大小
            min_count=0,  # 词频阈值
            sg=1,  # 使用Skip-gram算法
            hs=1,  # 使用Hierarchical Softmax
            workers=8,  # 线程数
        )
        model.wv.save_word2vec_format(output)  # 保存训练好的模型

# 主程序
if __name__ == "__main__":
    node_num = 2708  # 节点数量
    edge_txt = "dataset/cora_edges.txt"  # 边的txt文件路径
    G = PersonalizedRW(node_num, edge_txt, device='cuda')  # 创建PersonalizedRW对象，使用GPU
    path_len = 50  # 随机游走长度
    total_walks = G.build_corpus(path_len)  # 构建语料库
    G.train(total_walks, embed_size=64, window_size=3, output="embedding_result_data/RW/10轮10步64维.txt")  # 训练Word2Vec模型
    print("训练完成")
