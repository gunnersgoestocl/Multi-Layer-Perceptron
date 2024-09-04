#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "activator.h"
#include "loss_func.h"
#include "setting.h"

#ifndef _INIT_H_
#define _INIT_H_

/* 構造体```neuron```の宣言 */
typedef struct neuron_ neuron;

/* ニューロンの層を表す構造体```neuron_layer```の宣言 */
typedef struct neuron_layer_ neuron_layer;

/* 構造体```network_1layer```の宣言 */
typedef struct network_1layer_ network_1layer;

/* ニューロン層の初期化 */
neuron_layer *init_neuron_layer(int k, int N_k, int activator, int b_size);

/* ネットワーク層の線形リストの操作 */
network_1layer *rewind_network(network_1layer *neural_network);
network_1layer *push_network_back(neuron_layer *o_layer, neuron_layer *v_layer, network_1layer *neural_network);

/* メモリの解放 */
void free_neuron_layer(neuron_layer *layer);
void free_network_1layer(network_1layer *network);
void free_network(network_1layer *neural_network);

/* 順伝播、誤差逆伝播 */
network_1layer *forward_prop(config *setting, model *M, network_1layer *neural_network, double **x, double **y);
network_1layer *back_prop(config *setting, model *M, network_1layer *neural_network, double alpha, int b_size);

// ニューラルネットワークを表す構造体へのポインタ、ネットワークの格納する値をグラフに可視化して出力するファイルのポインタを引数に取る
void print_network(network_1layer *neural_network, FILE *fp_log);
void print_log_prop(config *setting, network_1layer *neural_network);

/* 構造体依存の活性化関数 */
void softmax(int start_idx, int v_class, neuron *neuron_1darray);

/* 学習済みのネットワークに対して、説明変数の値を入力して、目的変数の値を出力する関数 */
double **estimate(int b_size, int v_dim, network_1layer *neural_network);
void estimate_sub (int b_size, int iter_num, int v_dim, network_1layer *neural_network, double **y_hat);

#endif