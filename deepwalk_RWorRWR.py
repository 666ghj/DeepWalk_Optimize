from networkx.classes.function import nodes
import numpy as np
import os
import networkx as nx
import random
from tqdm import tqdm

from gensim.models import Word2Vec


class deepwalk_RWorRWR:
    # 初始化函数，输入节点数目，边的txt文件地址
    def __init__(self, node_num: int, edge_txt: str, undirected=False):
        print("初始化图结构")
        # 先判断是有向的还是无向的
        if undirected:
            self.G = nx.Graph()
        else:
            self.G = nx.DiGraph()
        # 生成一个输入num数目的节点列表，放在图中
        self.G.add_nodes_from(list(range(node_num)))

        # 节点解决了，开始解决边，我们去定义一个读取str格式的边信息的函数read_edge
        edges = self.read_edge(edge_txt)
        self.G.add_edges_from(edges)

        # 获取图信息
        self.adjacency = np.array(nx.adjacency_matrix(self.G).todense())
        self.G_neighbor = {}
        # 获取每个节点的邻居列表，然后构成一个邻居字典，字典的key就是这个源节点的node
        for i in range(self.adjacency.shape[0]):
            self.G_neighbor[i] = []
            for j in range(self.adjacency.shape[0]):
                if i == j:
                    continue
                if self.adjacency[i, j] > 0.01:
                    self.G_neighbor[i].append(j)


    # 输入一个边的txt文件地址，返回一个元组列表，每一个元组表示一条边
    def read_edge(self, edge_txt):
        # loadtxt非常适合读取由简单分隔符（如空格）分隔的数字
        edges = np.loadtxt(edge_txt, dtype=np.int16)
        # 创建一个元组列表，每一个元组表示一条边
        edges = [(edges[i, 0], edges[i, 1]) for i in range(edges.shape[0])]

        return edges


    # 生成一个 随机游走/重启随机游走 序列
    def RWorRWR(self, path_len, alpha=0, rand_iter=random.Random(628), start=None):
        """_summary_

        Args:
            path_len (_type_): 指定随机游走的长度
            alpha (int, optional): 重启的概率
            rand_iter (_type_, optional): 随机数生成器
            start (_type_, optional): 指定随机游走的起始节点
        """
        # 把初始化函数中的邻居字典给了G
        G = self.G_neighbor

        if start:
            rand_path = [start]
        else:
            # 没指定就随机选一个节点出来，随机概率与节点的任何属性都没有关系
            # G.keys()返回的是一个动态的字典键试图对象，我们给他转成列表
            rand_path = [rand_iter.choice(list(G.keys()))]

        # 开始一次随机游走
        while len(rand_path) < path_len:
            current_pos = rand_path[-1]

            if len(G[current_pos]) > 0:  # 当当前节点有邻节点时
                if rand_iter.random() > alpha:
                    rand_path.append(rand_iter.choice(G[current_pos]))
                else:  # 重启
                    rand_path.append(rand_path[0])
            else:  # 当碰到孤立节点以后
                break

        return [str(node) for node in rand_path]


    # 构建语料库
    def build_corpus(self, num_paths, path_len, alpha=0, rand_iter=random.Random(628)):
        """_summary_

        Args:
            num_paths (_type_): 决定生成多少组随机游走序列，一组包含全部节点的随机游走序列
            path_len (_type_): 指定随机游走的长度
            alpha (int, optional): 重启的概率
            rand_iter (_type_, optional): 随机数生成器
        """
        if alpha:
            print("开始生成重启随机游走语料库")
        else:
            print("开始生成随机游走语料库")

        total_walks = []
        G = self.G_neighbor
        nodes = list(G.keys())

        for i in tqdm(range(num_paths)):
            # 将节点列表打乱
            rand_iter.shuffle(nodes)
            # 分别从每个节点开始进行一次随机游走
            for node in nodes:
                total_walks.append(self.RWorRWR(path_len, alpha, rand_iter, start=node))

        return total_walks


    # 训练word2vec模型
    def train(self, total_walks, embed_size=64, window_size=3, output="."):
        """_summary_

        Args:
            total_walks (_type_): 语料库
            embed_size (int, optional): 词向量维度
            window_size (int, optional): 窗口大小
            output (str, optional): 输出路径
        """
        print("开始训练word2vec模型")
        model = Word2Vec(
            total_walks,
            vector_size=embed_size,
            window=window_size,
            min_count=0,  # 词频阈值，如果一个词的频率小于min_count，那么这个词将被忽略
            sg=1,  # 训练算法的选择，如果sg是0，那么将使用CBOW算法；如果sg是1，那么将使用Skip-gram算法
            hs=1,  # 训练模型时是否使用Hierarchical Softmax，如果hs是0，那么将使用Negative Sampling；如果hs是1，那么将使用Hierarchical Softmax
            workers=8,  # 训练模型时使用的线程数，表示在训练模型时，可以同时运行的线程数
        )

        model.wv.save_word2vec_format(output)


# 进行训练
if __name__ == "__main__":
    # 生成一个有向图
    node_num = 2707
    edge_txt = "dataset/cora_edges.txt"
    G = deepwalk_RWorRWR(node_num, edge_txt)

    # 生成语料库
    num_paths = 10
    path_len = 10
    total_walks = G.build_corpus(num_paths, path_len)
    # 训练word2vec模型
    G.train(total_walks, embed_size=64, window_size=3, output="embedding_result_data/RW/10轮10步64维.txt")
    
    print("训练完成")
