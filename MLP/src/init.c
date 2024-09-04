#include "init.h"
struct neuron_ {
  double v;
  double o;
  double delta;
  double delA;
};

struct neuron_layer_ {
  neuron **neuron_2darray;
  int k;
  int N_k;
  int activator;    // ACTIVATORのindexに対応
  int b_size;
};

struct network_1layer_ {
  double **weight;
  neuron_layer *o_layer;
  neuron_layer *v_layer;
  int o_dim;
  int v_dim;
  network_1layer *prev;
  network_1layer *next;
};


neuron_layer *init_neuron_layer(int k, int N_k, int activator, int b_size) {
    neuron_layer *layer = (neuron_layer *)malloc(sizeof(neuron_layer));
    layer -> k = k;
    layer -> N_k = N_k;
    layer -> activator = activator;
    layer -> b_size = b_size;
    layer -> neuron_2darray = (neuron **)malloc(sizeof(neuron *) * b_size);
    for (int i = 0; i < b_size; i++) {
        layer -> neuron_2darray[i] = (neuron *)malloc(sizeof(neuron) * N_k);
        for (int j = 0; j < N_k; j++) {
            layer -> neuron_2darray[i][j].v = 0.0;
            layer -> neuron_2darray[i][j].o = 0.0;
            layer -> neuron_2darray[i][j].delta = 0.0;
            layer -> neuron_2darray[i][j].delA = 0.0;
        }
    }
    return layer;
}


// 9.  誤差逆伝播をせずにポインタを末尾から先頭に戻す関数```rewind_network```を定義する
//     - 引数には、ネットワーク層構造体の線形リストの末尾へのポインタ```neural_network```をとる

network_1layer *rewind_network(network_1layer *neural_network) {
    while (neural_network -> prev != NULL) {
        neural_network = neural_network -> prev;
    }
    return neural_network;
}

// 4. ネットワーク層を表す構造体```network_1layer```の線形リストの末尾に、初期化された```network_1layer```構造体を追加する関数```push_network_back```を作る
//    - 引数には、ニューロン層を表す```neuron_layer```構造体を2つと線形リストの先頭へのポインタをとり、各層間の重みを $0$で 初期化した```network_1layer```構造体を追加する
//    - 最終的なリストの長さは $(D+1)$ となる
//    - 線形リストの名前は、```neural_network```とする

network_1layer *push_network_back(neuron_layer *o_layer, neuron_layer *v_layer, network_1layer *neural_network) {
    network_1layer *network = (network_1layer *)malloc(sizeof(network_1layer));
    network -> o_layer = o_layer;
    network -> v_layer = v_layer;
    network -> o_dim = o_layer -> N_k;
    network -> v_dim = v_layer -> N_k;
    network -> weight = (double **)malloc(sizeof(double *) * network -> o_dim);
    for (int i = 0; i < network -> o_dim; i++) {
        network -> weight[i] = (double *)malloc(sizeof(double) * network -> v_dim);
        for (int j = 0; j < network -> v_dim; j++) {
            // o_layer -> N_k に対して、平均が0 標準偏差が sqrt(2/N_k) の正規分布に従う乱数を発生させる
            network -> weight[i][j] = (double)rand() / RAND_MAX * 2.0 * sqrt(2.0 / (double)(o_layer -> N_k)) - sqrt(2.0 / (double)(o_layer -> N_k));
            // network -> weight[i][j] = 1.0;
        }
    }

    // 末尾のnetwork_1layerのnextに新たなnetwork_1layerを追加
    network_1layer *p = neural_network;
    // 線形リストが空の場合(最初ネットワーク層の場合)
    if (neural_network == NULL) {
        neural_network = network;
        return neural_network;
    }
    else {
        // 末尾に到達したい
        while (p->next != NULL) {
            p = p->next;    
        }
        p->next = network;    // 線形リストが延長された(NULLではなくなった)
        network -> prev = p;    // 双方向リストになるようにprevを更新
    }
    neural_network = rewind_network(neural_network);
    return neural_network;
}

// ニューロン層のメモリを解放
void free_neuron_layer(neuron_layer *layer) {
    for (int i = 0; i < layer -> b_size; i++) {
        free(layer -> neuron_2darray[i]);
    }
    free(layer -> neuron_2darray);
    free(layer);
}

// ネットワーク層構造体のメモリを解放
void free_network_1layer(network_1layer *network) {
    for (int i = 0; i < network -> o_dim; i++) {
        free(network -> weight[i]);
    }
    free(network -> weight);
    // free_neuron_layer(network -> o_layer);
    if (network -> v_layer -> k != 0) {
        free_neuron_layer(network -> v_layer);
    } else {
        free_neuron_layer(network -> v_layer);
        free_neuron_layer(network -> o_layer);
    }
    // free_neuron_layer(network -> v_layer);
    free(network);
}

// ネットワーク層構造体の線形リストのメモリを解放
void free_network(network_1layer *neural_network) {
    while (neural_network -> next != NULL) {
        neural_network = neural_network -> next;
    }
    while (neural_network -> prev != NULL) {
        neural_network = neural_network -> prev;
        free_network_1layer(neural_network -> next);
    }
    free_network_1layer(neural_network);
}


// 順伝播を行い、各ニューロンのメンバを更新して、推論を行い、最後のネットワーク層構造体のポインタを返す
network_1layer *forward_prop(config *setting, model *M, network_1layer *neural_network, double **x, double **y) {
    // 第0層の入力値vをxに設定
    for (int b = 0; b < (neural_network -> o_layer) -> b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            neural_network -> o_layer -> neuron_2darray[b][i].v = x[b][i];
        }
    }

    // 第0層の出力値oを更新
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            neural_network -> o_layer -> neuron_2darray[b][i].o = neural_network -> o_layer -> neuron_2darray[b][i].v;
        }
    }

    // 第1層から第D層までのニューロンの入力値と出力値を交互に計算し更新
    while (neural_network -> next != NULL) {
        
        // 第k層のニューロンの入力値vを計算 (線形変換)
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].v = 0.0;  // 初期化
                for (int j = 0; j < neural_network -> o_dim; j++) {
                    neural_network -> v_layer -> neuron_2darray[b][i].v += neural_network -> o_layer -> neuron_2darray[b][j].o * neural_network -> weight[j][i];
                }
            }
        }

        // 第k層のニューロンへの入力値vを活性化関数にかけて出力値oを計算
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].o = ACTIVATORS[neural_network -> v_layer -> activator](neural_network -> v_layer -> neuron_2darray[b][i].v);
            }
        }
        print_log_prop(setting, neural_network);

        // 第k層のニューロンの活性化関数の勾配delAを計算
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].delA = ACTIVATORS_GRAD[neural_network -> v_layer -> activator](neural_network -> v_layer -> neuron_2darray[b][i].v);
            }
        }
        print_log_prop(setting, neural_network);

        // 次のネットワーク層へ
        neural_network = neural_network -> next;
    }

    // 第D+1層のニューロンの入力値vを計算
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> v_dim; i++) {
            neural_network -> v_layer -> neuron_2darray[b][i].v = 0.0;  // 初期化
            for (int j = 0; j < neural_network -> o_dim; j++) {
                neural_network -> v_layer -> neuron_2darray[b][i].v += (neural_network -> o_layer -> neuron_2darray[b][j].o) * (neural_network -> weight[j][i]);
            }
        }
    }
    print_log_prop(setting, neural_network);

    // 第D+1層のニューロンの出力値oを計算、出力層のactivatorにより変わる
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        // 回帰もしくは二値分類の場合
        if (M -> out_activator < 10) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].o = ACTIVATORS[M -> out_activator](neural_network -> v_layer -> neuron_2darray[b][i].v);
            }
        }
        // 多クラス分類の場合
        else if (M -> out_activator == 10) {
            int idx = 0;    // 各ラベルの出力値の開始インデックス (eg. 性別:0=<idx<v_class[0])
            for (int i = 0; i < neural_network -> v_dim; i++) {
                softmax(idx, M -> v_class[i], neural_network -> v_layer -> neuron_2darray[b]);
                idx += M -> v_class[i];
            }
        }
    }
    print_log_prop(setting, neural_network);

    // 第D+1層(出力層)のニューロンのdeltaを計算、linear&MSE or softmax&cross_entropy では以下の計算
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> v_dim; i++) {
            int out_delta_func = get_value(M -> out_activator, setting -> loss_func);
            // neural_network -> v_layer -> neuron_2darray[b][i].delta = neural_network -> v_layer -> neuron_2darray[b][i].o - y[b][i];
            neural_network -> v_layer -> neuron_2darray[b][i].delta = OUT_DELTA_FUNC[out_delta_func](neural_network -> v_layer -> neuron_2darray[b][i].o, y[b][i]);
        }
    }
    print_log_prop(setting, neural_network);

    return neural_network;  // 線形リストの末尾へのポインタを返す
}

// 誤差逆伝播を行い、ニューロンおよびネットワークのメンバを更新して、先頭の(最も入力層側の)ネットワーク層構造体のポインタを返す
network_1layer *back_prop(config *setting, model *M, network_1layer *neural_network, double alpha, int b_size) {
    if (setting -> show_values == 1 || setting -> program_log == 1) {
        fprintf(setting -> fp_log, "\n===============<<back_prop>>================\n");
    }

    // 第D+1層の行列成分Uを更新
    for (int i = 0; i < neural_network -> o_dim; i++) {
        for (int j = 0; j < neural_network -> v_dim; j++) {
            double del_Uij = 0.0;
            for (int b = 0; b < b_size; b++) {
                del_Uij += (neural_network -> v_layer -> neuron_2darray[b][j].delta) * (neural_network -> o_layer -> neuron_2darray[b][i].o);
            }
            neural_network -> weight[i][j] -= alpha * del_Uij / b_size;
            // 浮動小数点のアンダーフローを防ぐため、0に近い値は0にする
            if (fabs(neural_network -> weight[i][j]) <= 0.00001) {
                neural_network -> weight[i][j] = 0.0;
            }
        }
    }
    print_log_prop(setting, neural_network);

    // 第D層のdeltaを更新
    for (int b = 0; b < b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            double delta_sum = 0.0;
            for (int l = 0; l < neural_network -> v_dim; l++) {
                delta_sum += (neural_network -> v_layer -> neuron_2darray[b][l].delta) * (neural_network -> weight[i][l]);
            }
            (neural_network -> o_layer -> neuron_2darray[b][i].delta) = delta_sum * (neural_network -> o_layer -> neuron_2darray[b][i].delA);
            // 浮動小数点のアンダーフローを防ぐため、0に近い値は0にする
            if (fabs(neural_network -> o_layer -> neuron_2darray[b][i].delta) <= 0.00001) {
                (neural_network -> o_layer -> neuron_2darray[b][i].delta) = 0.0;
            }
        }
    }
    print_log_prop(setting, neural_network);

    // 第D層から第2層まで、ネットワーク層の行列成分Uとo_layerのdeltaを交互に更新
    while (neural_network -> o_layer -> k > 1) {
        neural_network = neural_network -> prev;
        print_log_prop(setting, neural_network);

        // 第kネットワーク層の行列成分Uを更新
        for (int i = 0; i < neural_network -> o_dim; i++) {
            for (int j = 0; j < neural_network -> v_dim; j++) {
                double del_Uij = 0.0;
                for (int b = 0; b < b_size; b++) {
                    del_Uij += (neural_network -> v_layer -> neuron_2darray[b][j].delta) * (neural_network -> o_layer -> neuron_2darray[b][i].o);
                }
                neural_network -> weight[i][j] -= alpha * del_Uij / b_size;
                // 浮動小数点のオーバーフローを防ぐため、0に近い値は0にする
                if (fabs(neural_network -> weight[i][j]) <= 0.00001) {
                    neural_network -> weight[i][j] = 0.0;
                } else if (fabs(neural_network -> weight[i][j]) >= 10000) {
                    fprintf(setting -> fp_log, "U of the %dth layer is too large\n", neural_network -> v_layer -> k);
                }
            }
        }
        print_log_prop(setting, neural_network);

        // 第kネットワーク層のo_layerのdeltaを更新
        for (int b = 0; b < b_size; b++) {
            for (int i = 0; i < neural_network -> o_dim; i++) {
                double delta_sum = 0.0;
                for (int l = 0; l < neural_network -> v_dim; l++) {
                    delta_sum += (neural_network -> v_layer -> neuron_2darray[b][l].delta) * (neural_network -> weight[i][l]);
                }
                (neural_network -> o_layer -> neuron_2darray[b][i].delta) = delta_sum * (neural_network -> o_layer -> neuron_2darray[b][i].delA);
                // 浮動小数点のアンダーフローを防ぐため、0に近い値は0にする、オーバーフローはエラーとして終了
                if (fabs(neural_network -> o_layer -> neuron_2darray[b][i].delta) <= 0.00001) {
                    (neural_network -> o_layer -> neuron_2darray[b][i].delta) = 0.0;
                } else if (fabs(neural_network -> o_layer -> neuron_2darray[b][i].delta) >= 1000000) {
                    fprintf(setting -> fp_log, "delta of the %dth layer is too large. Please re-design neural network or learning rate!\n", neural_network -> o_layer -> k);
                    exit(1);
                }
            }
        }
        print_log_prop(setting, neural_network);
    }
    print_log_prop(setting, neural_network);
    neural_network = neural_network -> prev;

    // 第1ネットワーク層の行列成分Uを更新
    for (int i = 0; i < neural_network -> o_dim; i++) {
        for (int j = 0; j < neural_network -> v_dim; j++) {
            double del_Uij = 0.0;
            for (int b = 0; b < b_size; b++) {
                del_Uij += (neural_network -> v_layer -> neuron_2darray[b][j].delta) * (neural_network -> o_layer -> neuron_2darray[b][i].o);
            }
            neural_network -> weight[i][j] -= alpha * del_Uij / b_size;
        }
    }
    print_log_prop(setting, neural_network);

    return neural_network;
}


// neural_network 構造体をグラフとして図示する関数
// k番目の neuron_layer の、ニューロンへの入力値とニューロンからの出力値を表示する
// k番目の network_layer の、ニューロン間の重みをテーブルで表示する
// 以上を繰り返すことで、全体の構造を表示する
void print_network(network_1layer *neural_network, FILE *fp_log){

    // 線形リストの先頭から末尾まで走破する
    int i = 0;
    // 線形リストの先頭に戻す
    network_1layer *p = rewind_network(neural_network);
    while(p != NULL){
        fprintf(fp_log, "[neuron layer %d (dim = %d)]\n", i, p->o_dim);

        // ニューロン層への入力値を表示
        fprintf(fp_log, "-------input value--------\n");
        //int j = 0;
        for (int j = 0; j < p->o_dim; j++){
            int k = 0;
            for (k = 0; k < p->o_layer->b_size;k++){
                fprintf(fp_log, "%4.4f ", (p->o_layer->neuron_2darray)[k][j].v);
            }
            fprintf(fp_log, "\n");
        }
        fprintf(fp_log, "\n");

        // ニューロン層からの出力値を表示
        fprintf(fp_log, "-------output value (after activation)--------\n");
        //int j = 0;
        for (int j = 0; j < p->o_dim; j++){
            int k = 0;
            for (k = 0; k < p->o_layer->b_size;k++){
                fprintf(fp_log, "%4.4f ", (p->o_layer->neuron_2darray)[k][j].o);
            }
            fprintf(fp_log, "\n");
        }
        fprintf(fp_log, "\n");

        // ニューロン層のdelAを表示
        fprintf(fp_log, "-------delA (gradient of activator)--------\n");
        //int j = 0;
        for (int j = 0; j < p->o_dim; j++){
            int k = 0;
            for (k = 0; k < p->o_layer->b_size;k++){
                fprintf(fp_log, "%4.4f ", (p->o_layer->neuron_2darray)[k][j].delA);
            }
            fprintf(fp_log, "\n");
        }
        fprintf(fp_log, "\n");

        // ニューロン層のdeltaを表示
        fprintf(fp_log, "-------delta --------\n");
        for (int j = 0; j < p->o_dim; j++){
            int k = 0;
            for (k = 0; k < p->o_layer->b_size;k++){
                fprintf(fp_log, "%4.4f ", (p->o_layer->neuron_2darray)[k][j].delta);
            }
            fprintf(fp_log, "\n");
        }
        fprintf(fp_log, "\n");
        
        // ニューロン間の重みを表示
        fprintf(fp_log, "-------weight (neuron_layer %d to %d)--------\n", i, i+1);
        int j_o = 0;
        int j_v = 0;
        for (j_o = 0; j_o < p->o_dim; j_o++){
            for (j_v = 0; j_v < p->v_dim; j_v++){
                fprintf(fp_log, "%4.6f ", p->weight[j_o][j_v]);
            }
            fprintf(fp_log, "\n");
        }
        fprintf(fp_log, "\n");
        // print_layer(p, fp_log);
        // 出力層の手前のネットワーク層に到達したら、出力層も表示して終了
        if (p->next == NULL){
            // break;
            fprintf(fp_log, "[neuron layer %d (dim = %d)]\n", i+1, p->v_dim);
            // ニューロン層への入力値を表示
            fprintf(fp_log, "-------input value--------\n");
            //int j = 0;
            for (int j = 0; j < p->v_dim; j++){
                int k = 0;
                for (k = 0; k < p->v_layer->b_size;k++){
                    fprintf(fp_log, "%4.4f ", (p->v_layer->neuron_2darray)[k][j].v);
                }
                fprintf(fp_log, "\n");
            }
            fprintf(fp_log, "\n");
            // ニューロン層からの出力値を表示
            fprintf(fp_log, "-------output value (after activation)--------\n");
            //int j = 0;
            for (int j = 0; j < p->v_dim; j++){
                int k = 0;
                for (k = 0; k < p->v_layer->b_size;k++){
                    fprintf(fp_log, "%4.4f ", (p->v_layer->neuron_2darray)[k][j].o);
                }
                fprintf(fp_log, "\n");
            }
            fprintf(fp_log, "\n");
            // ニューロン層のdelAを表示
            fprintf(fp_log, "-------delA (gradient of activator)--------\n");
            //int j = 0;
            for (int j = 0; j < p->v_dim; j++){
                int k = 0;
                for (k = 0; k < p->v_layer->b_size;k++){
                    fprintf(fp_log, "%4.4f ", (p->v_layer->neuron_2darray)[k][j].delA);
                }
                fprintf(fp_log, "\n");
            }
            fprintf(fp_log, "\n");
            // ニューロン層のdeltaを表示
            fprintf(fp_log, "-------delta --------\n");
            for (int j = 0; j < p->v_dim; j++){
                int k = 0;
                for (k = 0; k < p->v_layer->b_size;k++){
                    fprintf(fp_log, "%4.4f ", (p->v_layer->neuron_2darray)[k][j].delta);
                }
                fprintf(fp_log, "\n");
            }
            fprintf(fp_log, "\n");

            // fprintf(fp_log, "\n");
            fprintf(fp_log,"\n==========----------------------------==========\n\n");
            break;
        }
        else{
            p = p->next;
            i++;
        }
    }

}

// propagation の途中経過を表示する関数
void print_log_prop(config *setting, network_1layer *neural_network){
    if (setting -> show_values == 1){
        fprintf(setting -> fp_log, "neural_network is at %dth layer.\n", neural_network -> o_layer -> k);
        print_network(neural_network, setting -> fp_log);
    } else if (setting -> program_log == 1){
        fprintf(setting -> fp_log, "neural_network is at %dth layer.\n", neural_network -> o_layer -> k);
    }
}

// 多クラス分類用のソフトマックス関数
void softmax(int start_idx, int v_class, neuron *neuron_1darray){
    double sum = 0;
    for (int i = start_idx; i < v_class; i++){
        sum += exp(neuron_1darray[i].v);
    }
    for (int i = start_idx; i < v_class; i++){
        neuron_1darray[i].o = exp(neuron_1darray[i].v) / sum;
    }
}

double **estimate(int b_size, int v_dim, network_1layer *neural_network) {
    // 出力の推定量を二次元配列に格納する
    double **y_hat = (double **)malloc(sizeof(double *) * b_size);
    for (int b = 0; b < b_size; b++) {
        y_hat[b] = (double *)malloc(sizeof(double) * v_dim);
        for (int i = 0; i < v_dim; i++) {
            y_hat[b][i] = neural_network -> v_layer -> neuron_2darray[b][i].o;
        }
    }
    return y_hat;
}

void estimate_sub (int b_size, int iter_num, int v_dim, network_1layer *neural_network, double **y_hat) {
    for (int b = 0; b < b_size; b++) {
        for (int i = 0; i < v_dim; i++) {
            y_hat[iter_num*b_size+b][i] = neural_network -> v_layer -> neuron_2darray[b][i].o;
        }
    }
}