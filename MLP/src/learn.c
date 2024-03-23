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

void load_testData(FILE *fp, int *test_flag, int b_size, int test_iter, double **x_test, double **y_test, int o_dim, int *o_col, int v_dim, int *v_col){
    // int col = 0;    // 列番号のカーソル // 変数名捜索のためのcolは関数内に格納するため、外出しにする
    // char buf[256];  // ファイルの1行を読み込むためのバッファ
    // char *token;   // 上の方の定義は関数内に格納するため、ここで定義する
    if (*test_flag == 0) {
        // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
        load_Data(fp, b_size*test_iter, x_test, y_test, o_dim, o_col, v_dim, v_col);    // 1バッチのデータ読み込み完了
        
        // if (program_confirmation == 1) {
        //     fprintf(fp_log, "The test data has been loaded.\n");
        // }
        *test_flag = 1;
    }
}

void load_Data(FILE *fp, int b_size, double **x, double **y, int o_dim, int *o_col, int v_dim, int *v_col){
    int col = 0;    // 列番号のカーソル // 変数名捜索のためのcolは関数内に格納するため、外出しにする
    char buf[256];  // ファイルの1行を読み込むためのバッファ
    char *token;   // 上の方の定義は関数内に格納するため、ここで定義する
    // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に格納する
    for (int b = 0; b < b_size; b++) {
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;
        while (token != NULL) {
            for (int i = 0; i < o_dim; i++) {
                if (col == o_col[i]) {
                    x[b][i] = atof(token);
                }
            }
            for (int i = 0; i < v_dim; i++) {
                if (col == v_col[i]) {
                    y[b][i] = atof(token);
                }
            }
            // if (col == v_col) {
            //     y[b][0] = atof(token);
            // }
            token = strtok(NULL, ";\n");
            col++;
        }
    }   // 1バッチのデータ読み込み完了
}