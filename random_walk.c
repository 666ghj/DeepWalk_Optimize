// gcc -shared -o random_walk.so -fPIC random_walk.c, 生成动态链接库

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 生成单次随机游走序列的函数
void random_walk_from_matrix(float* matrix, int* walk, int start, int path_len, int n, unsigned int seed) {
    srand(seed);
    walk[0] = start;
    int current_node = start;

    for (int i = 1; i < path_len; ++i) {
        float weights_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            weights_sum += matrix[current_node * n + j];
        }

        if (weights_sum == 0.0) {
            break;
        }

        float r = ((float) rand() / (float) RAND_MAX) * weights_sum;
        float cumulative_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            cumulative_sum += matrix[current_node * n + j];
            if (cumulative_sum >= r) {
                walk[i] = j;
                current_node = j;
                break;
            }
        }
    }
}

// 生成所有节点随机游走序列的函数
void build_corpus(float* matrix, int* walks, int num_nodes, int path_len, int num_walks, int n) {
    int total_steps = num_walks * num_nodes;
    for (int w = 0; w < num_walks; ++w) {
        for (int start_node = 0; start_node < num_nodes; ++start_node) {
            unsigned int seed = (unsigned int) time(NULL) + w * num_nodes + start_node;
            random_walk_from_matrix(matrix, walks + (w * num_nodes + start_node) * path_len, start_node, path_len, n, seed);

            // 计算进度并显示
            int step = w * num_nodes + start_node + 1;
            int progress = (int) ((float) step / total_steps * 100);
            if (step % (total_steps / 100) == 0) {
                printf("\rProgress: %d%%", progress);
                fflush(stdout);
            }
        }
    }
    printf("\n");
}
