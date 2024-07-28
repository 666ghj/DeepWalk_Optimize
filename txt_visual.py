import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 通过deepwalk生成的txt嵌入文件进行可视化

# 读取txt文件
# 读取第一行作为头部或特殊数据处理
header = pd.read_csv('embedding_result_data/RWR/重启0.4_100轮10步128维.txt', sep=" ", header=None, nrows=1)
print("节点个数：{}, 节点嵌入维度：{}".format(header[0][0], header[1][0]))
# 从第二行开始读取其余数据
data = pd.read_csv('embedding_result_data/RWR/重启0.4_100轮10步128维.txt', sep=" ", header=None, skiprows=1)
data = data.drop(0, axis=1)
print(data)

# 使用PCA降维data数据
pca = PCA(n_components=2)
result = pca.fit_transform(data)
print("降维后的数据：")
print(result)

# 可视化
plt.figure(figsize=(16, 12))
plt.scatter(result[:, 0], result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Embedding')
plt.show()