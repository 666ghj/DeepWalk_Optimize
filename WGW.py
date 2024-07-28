import random
from line_for_tencent import get_ss
from target_line import get_tt
from visdom import Visdom
from test import get_source_vector
from target import get_target_vector
import torch.nn.functional as F
from mapping_model import PaleMappingMlp
from sklearn.manifold import TSNE
import numpy as np
import torch
import time
star_time = time.time()
from evaluate import get_statistics
import math
import os
from sequence_counter import count_sequences, count_graph_properties
from get_data import get_edges_groundtruth
from get_stg import get_edges_groundtruth
from basic_model import get_source_embedding
from target_basic import get_target_embedding

from urp_source import urp_source_vector
from urp_target import urp_target_vector
import visdom

os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'
np.random.seed(616)
torch.manual_seed(616) # 为CPU设置随机种子
torch.cuda.manual_seed(616) # 为当前GPU设置随机种子


embedding_dim = 128
# source_vector,anchor_id2idx = get_source_vector()
# source_vector = np.genfromtxt('source_embedding_ur_every_nonom.txt')
# target_vector = np.genfromtxt('target_embedding_ur_every_nonom.txt')

# source_vector = get_source_embedding()
# target_vector = get_target_embedding()
# source_vector1 = get_source_vector()

# target_vector1 = get_target_vector()
#source_vector2 = get_ss()
#target_vector2 = get_tt()
source_vector = urp_source_vector()
target_vector = urp_target_vector()

#source_vector = source_vector1
#target_vector = target_vector1
#source_vector = np.concatenate((source_vector1,source_vector2),axis=1)


source_vector = torch.FloatTensor(source_vector)
source_vector = F.normalize(source_vector,dim=1)
source_vector = source_vector.numpy()
#
#target_vector = np.concatenate((target_vector1,target_vector2),axis=1)
#

target_vector = torch.FloatTensor(target_vector)
target_vector = F.normalize(target_vector,dim=1)
target_vector = target_vector.numpy()

# source_vector = 3*source_vector1+5*source_vector2
# target_vector = 3*target_vector1+5*target_vector2


# source_vector = torch.FloatTensor(source_vector)
# source_vector = F.normalize(source_vector,dim=1)
# source_vector = source_vector.numpy()
#
# target_vector = torch.FloatTensor(target_vector)
# target_vector = F.normalize(target_vector,dim=1)
# target_vector = target_vector.numpy()


# source_vector = (source_vector1+source_vector2)/2
# target_vector = (target_vector1+target_vector2)/2
map_act = 'relu'
map_lr = 0.001
map_batch_size = 32
source_vector = torch.FloatTensor(source_vector)
target_vector = torch.FloatTensor(target_vector)
source_vector = source_vector.cuda()
target_vector = target_vector.cuda()
viz = visdom.Visdom()
print("Use Mpl mapping")
mapping_model = PaleMappingMlp(
    embedding_dim=embedding_dim,
    source_embedding=source_vector,
    target_embedding=target_vector,
    activate_function=map_act,
)

mapping_model = mapping_model.cuda()
optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, mapping_model.parameters()), lr=map_lr)

source_edges,target_edges,groundtruth = get_edges_groundtruth()
# groundtruth = np.genfromtxt("../tencent/groundtruth", dtype=np.int32)
np.random.shuffle(groundtruth)
# np.savetxt("../dataspace/shuffle_ground.txt", groundtruth, fmt="%d", delimiter=" ")

# groundtruth = np.genfromtxt("../dataspace/shuffle_ground.txt", dtype=np.int32)
groundtruth = groundtruth.tolist()

# for i in groundtruth:
#     i[0] = anchor_id2idx.get(i[0])
#
# print(groundtruth)

groundtruth_source = []
groundtruth_target = []
groundtruth_dict = {}
id2idx_source = {}
id2idx_target = {}
for i in groundtruth:
    groundtruth_source.append(i[0])
    groundtruth_target.append(i[1])
    groundtruth_dict[i[0]] = i[1]
for index,value in enumerate(groundtruth):
    id2idx_source[value[0]] = index
    id2idx_target[value[1]] = index
train_size = math.floor(0.4*len(groundtruth_source))
val_size = math.floor(0.1*len(groundtruth_source))
source_train_nodes = groundtruth_source[:train_size]
source_val_nodes = groundtruth_source[train_size:train_size+val_size]
source_test_nodes = groundtruth_source[train_size:]


n_iters = len(source_train_nodes) // map_batch_size
assert n_iters > 0, "batch_size is too large"
if (len(source_train_nodes) % map_batch_size > 0):
    n_iters += 1
print_every = int(n_iters / 4) + 1
total_steps = 0



for epoch in range(1,200):
    # np.random.shuffle(source_t    rain_nodes)
    start = time.time()
    print('Epochs: ', epoch)
    for iter in range(n_iters):
        source_batch = source_train_nodes[iter * map_batch_size:(iter + 1) * map_batch_size]
        target_batch = [groundtruth_dict[x] for x in source_batch]

        
        source_batch = torch.LongTensor(source_batch)
        target_batch = torch.LongTensor(target_batch)
        source_batch = source_batch.cuda()
        target_batch = target_batch.cuda()
        optimizer.zero_grad()
        start_time = time.time()

        loss = mapping_model.loss(source_batch, target_batch)
        loss.backward()
        optimizer.step()

        if total_steps % print_every == 0 and total_steps > 0:
            print("Iter:", '%03d' % iter,
                  "train_loss=", "{:.5f}".format(loss.item()),
                  "time", "{:.5f}".format(time.time() - start_time)
                                    )
        total_steps += 1
source_after_mapping = mapping_model(source_vector)
S = torch.matmul(source_after_mapping,target_vector.t())
S = S.detach().cpu().numpy()

source_after_mapping = source_after_mapping.detach().cpu().numpy()
target_vector = target_vector.cpu().numpy()
# s_down = TSNE(n_components=2).fit_transform(source_after_mapping)
# t_down = TSNE(n_components=2).fit_transform(target_vector)
#
# s_anchor = []
# for i in groundtruth:
#     s_anchor.append(i[0])
#
# anchor_list = []
# for i in range(5313):
#     if i in s_anchor:
#         anchor_list.append(1)
#     else:
#         anchor_list.append(2)
#
# x = []
# y = []
# for i in source_after_mapping:
#     x.append(i[0])
# x = np.array(x)
# y = np.array(anchor_list)
#
# color = np.array([[47,16,244],[237,26,26]])
# win = viz.scatter(
#     X=s_down,
#     Y=y,
#     opts=dict(
#         markersize=10,
#         markersymbol='cross-thin-open',
#         markercolor=color,
#         legend=['1', '2'],
#     ),
# )
# z = []
# for i in range(5120):
#     z.append(1)
# z = np.array(z)
# win = viz.scatter(
#     X=t_down,
#     Y=z,
#     opts=dict(
#         markersize=10,
#         markersymbol='cross-thin-open',
#     ),
# )
#
# print(S.shape)


gt = np.zeros((3906,1118))
gt2 = np.zeros((3906,1118))
#gt = np.zeros((3493,1792))
#gt2 = np.zeros((3493,1792))
#print(source_test_nodes)
for i in source_train_nodes:
    gt[i,groundtruth_dict[i]] = 1

for i in source_test_nodes:
    gt2[i,groundtruth_dict[i]] = 1

get_statistics(S, gt)
get_statistics(S,gt2)

end_time = time.time()
execution_time = end_time - star_time

print("Time:",round(execution_time,2))

source_path = "../data/source.txt"
target_path = "../data/target.txt"

#source_size = os.path.getsize(source_path)/1024
#target_size = os.path.getsize(target_path)/1024

source_sequences = count_sequences(source_path)
target_sequences = count_sequences(target_path)

num_source_sequences, num_source_edges, num_source_nodes = count_graph_properties(source_sequences)
num_target_sequences, num_target_edges, num_target_nodes = count_graph_properties(target_sequences)

num_sequences = num_source_sequences + num_target_sequences
num_edges = num_source_edges + num_target_edges
num_nodes = num_source_nodes + num_target_nodes

print("序列总数:", num_sequences)
print("边的总数:", num_edges)
print("节点总数:", num_nodes)

#print('source_size:',source_size)
#print('target_size:',target_size)

