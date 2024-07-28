import numpy as np
import matplotlib.pyplot as plt

# 加载npy文件
adj_matrix = np.load('dataset/cora_adj.npy')

# 找到所有节点的位置
y, x = np.where(adj_matrix > 0)

# 绘制邻接矩阵
plt.figure(figsize=(16, 12))
plt.imshow(adj_matrix, cmap='binary')

# 使用scatter函数绘制节点
plt.scatter(x, y, color='red')

plt.title('Adjacency Matrix')
plt.show()