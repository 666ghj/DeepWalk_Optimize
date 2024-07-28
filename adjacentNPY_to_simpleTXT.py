import numpy as np
from tqdm import tqdm

# 该程序实现将 npy格式的邻接矩阵 转换为 txt格式的有向图或无向图的边列表

def adj_matrix_to_edge_list_directed(adj_matrix, output_file):
    # 打开一个新的.txt文件，准备写入边列表
    with open(output_file, 'w') as f:
        # 遍历邻接矩阵的每一个元素
        for i in tqdm(range(adj_matrix.shape[0]), desc="Processing directed graph"):
            for j in range(adj_matrix.shape[1]):
                # 如果这两个节点之间有边
                if adj_matrix[i, j] != 0:
                    # 将这条边写入.txt文件
                    f.write(f'{i} {j}\n')

def adj_matrix_to_edge_list_undirected(adj_matrix, output_file):
    # 打开一个新的.txt文件，准备写入边列表
    with open(output_file, 'w') as f:
        # 遍历邻接矩阵的上三角部分
        for i in tqdm(range(adj_matrix.shape[0]), desc="Processing undirected graph"):
            for j in range(i+1, adj_matrix.shape[1]):
                # 如果这两个节点之间有边
                if adj_matrix[i, j] != 0:
                    # 将这条边写入.txt文件
                    f.write(f'{i} {j}\n')
                    
                    
# 使用样例
# # 从.npy文件加载邻接矩阵
# adj_matrix = np.load('adj_matrix.npy')

# # 将邻接矩阵转换为有向图的边列表
# adj_matrix_to_edge_list_directed(adj_matrix, 'edge_list_directed.txt')

# # 将邻接矩阵转换为无向图的边列表
# adj_matrix_to_edge_list_undirected(adj_matrix, 'edge_list_undirected.txt')

adj_matrix = np.load('dataset/cora_adj.npy')
adj_matrix_to_edge_list_undirected(adj_matrix, 'dataset/cora_edges.txt')