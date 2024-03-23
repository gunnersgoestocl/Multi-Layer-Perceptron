#include "learn.h"

network_1layer  *init_network(int o_dim, int v_dim, int b_size, int D, int *N, FILE *fp_log, int program_confirmation){
    // ニューラルネットワークの層を表す構造体の線形リストの先頭へのポインタを作る
    network_1layer *neural_network = NULL;
    // ニューラルネットワークの入力層を表すニューロン層を作る
    neuron_layer *o_layer = init_neuron_layer(0, o_dim, b_size);
    // ニューラルネットワークの中間層を表すニューロン層およびネットワーク層の線形リストを作る
    for (int i=0; i < D; i++){
        neuron_layer *v_layer = init_neuron_layer(i+1, N[i], b_size);
        neural_network = push_network_back(o_layer, v_layer, neural_network);
        o_layer = v_layer;
    }
    // ニューラルネットワークの出力層を表すニューロン層を作る
    neuron_layer *v_layer = init_neuron_layer(D+1, v_dim, b_size);
    neural_network = push_network_back(o_layer, v_layer, neural_network);    // 第 D+1 層まで初期化されたニューラルネットワークを表す構造体の線形リストができた
    if (program_confirmation == 1) {
        fprintf(fp_log, "The neural network has been initialized.\n");
    }
    return neural_network;
    // fprintf(fp_log, "The neural network has been initialized.\n");
    // ここまでエラーなし 12/30 22:25
}