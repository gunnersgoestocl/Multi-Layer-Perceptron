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


neuron_layer *init_neuron_layer(int k, int N_k, int b_size) {
    neuron_layer *layer = (neuron_layer *)malloc(sizeof(neuron_layer));
    layer -> k = k;
    layer -> N_k = N_k;
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

// 10. ニューラルネットワークのメモリを解放する関数```free_network```を定義する
//     - 引数には、ネットワーク層構造体の線形リストの先頭へのポインタ```neural_network```をとる
//     - 先頭から末尾に向かって、各ネットワーク層構造体のメモリを解放する
//     - その後、ネットワーク層構造体の線形リストのメモリを解放する
//     - それぞれ、各ニューロン層構造体のメモリを解放する関数```free_neuron_layer```と、各ネットワーク層構造体のメモリを解放する関数```free_network_1layer```を定義する

void free_neuron_layer(neuron_layer *layer) {
    for (int i = 0; i < layer -> b_size; i++) {
        free(layer -> neuron_2darray[i]);
    }
    free(layer -> neuron_2darray);
    free(layer);
}

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


// 5. 関数```forward_prop```を定義する (void型)
//   - 出力の推定量を求める
//   - 説明変数値の二次元配列```x```と、目的変数値の二次元配列```y```と、ネットワーク層構造体の線形リストの先頭へのポインタ```neural_network```と、活性化関数のポインタ```activator```を引数にとり、各ニューロンの入力値```v```と出力値```o```などを更新していく
//   - 手順は、
//     - 第 $0$ ニューロン層の第 $b$ 行の入力値```v```を```x[b]```(これはミニバッチの $b$番目のデータに相当)の各要素に設定する
//       - ということを、$b$について $0<=b<b_size$ で繰り返す
//     - 第 $0$ ニューロン層の出力値```o```を更新する(この時は```v```と```o```は同じ値)
//     - 第 $D$ 層までのニューロンの入力値と出力値を交互に計算し更新する(行列の積を計算する)
//       - この時、第 $k$ 層のニューロンの入力値```v```は、第 $k-1$ 層のニューロンの出力値```o```と、第 $k$ 層の重み```weight```の行列の積として計算する
//       - 第 $k$ 層のニューロンへの入力値```v```を```activator```(leakly_relu)にかけ、第 $k$ 層のニューロンの出力値```o```を計算し更新する
//       - ```v```の更新後、活性化関数の勾配```delA```をバッチ・ニューロンごとに計算し更新する
//     - 第 $D+1$ ニューロン層への入力値```v```から第 $D+1$ ニューロン層の出力値```o```を返す(この時は```v```と```o```は同じ値でこれが出力の推定量となる)
//       - 代わりに、目的変数値のベクトル```y```を元に、第 $D+1$ ニューロン層の```delta```に、各データごと($0 <= b < b_size$)に```o```と```y```の差をとったものを代入する

network_1layer *forward_prop(FILE *fp_log, double **x, double **y, network_1layer *neural_network, double (*activator)(double), double (*activator_grad)(double)) {

    fprintf(fp_log,"\n==============<<forward_prop>>=================\n");
    // 第0層の入力値vをxに設定
    for (int b = 0; b < (neural_network -> o_layer) -> b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            neural_network -> o_layer -> neuron_2darray[b][i].v = x[b][i];
        }
    }
    fprintf(fp_log,"setting v of the 0th layer is completed\n");
    print_network(neural_network, fp_log);

    // 第0層の出力値oを更新
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            neural_network -> o_layer -> neuron_2darray[b][i].o = neural_network -> o_layer -> neuron_2darray[b][i].v;
        }
    }
    fprintf(fp_log,"updating o of the 0th layer is completed\n");
    print_network(neural_network, fp_log);

    // 第1層から第D層までのニューロンの入力値と出力値を交互に計算し更新
    while (neural_network -> next != NULL) {
        
        // 第k層のニューロンの入力値vを計算
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].v = 0.0;  // 初期化
                for (int j = 0; j < neural_network -> o_dim; j++) {
                    neural_network -> v_layer -> neuron_2darray[b][i].v += neural_network -> o_layer -> neuron_2darray[b][j].o * neural_network -> weight[j][i];
                }
            }
        }
        fprintf(fp_log, "calculating v of the %dth layer is completed\n", neural_network -> v_layer -> k);
        print_network(neural_network, fp_log);

        // 第k層のニューロンの活性化関数の勾配delAを計算
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].delA = (*activator_grad)(neural_network -> v_layer -> neuron_2darray[b][i].v);
            }
        }
        fprintf(fp_log, "calculating delA of the %dth layer is completed\n", neural_network -> v_layer -> k);
        print_network(neural_network, fp_log);

        // 第k層のニューロンへの入力値vを活性化関数にかけて出力値oを計算
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].o = (*activator)(neural_network -> v_layer -> neuron_2darray[b][i].v);
            }
        }
        fprintf(fp_log, "updating o of the %dth layer is completed\n", neural_network -> v_layer -> k);
        print_network(neural_network, fp_log);

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
    fprintf(fp_log, "calculating v of the %dth layer is completed\n", neural_network -> v_layer -> k);
    print_network(neural_network, fp_log);

    // 第D+1層のニューロンの出力値oを計算
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> v_dim; i++) {
            neural_network -> v_layer -> neuron_2darray[b][i].o = neural_network -> v_layer -> neuron_2darray[b][i].v;
        }
    }
    fprintf(fp_log, "updating o of the %dth layer is completed\n", neural_network -> v_layer -> k);
    print_network(neural_network, fp_log);

    // 第D+1層のニューロンのdeltaを計算
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> v_dim; i++) {
            neural_network -> v_layer -> neuron_2darray[b][i].delta = neural_network -> v_layer -> neuron_2darray[b][i].o - y[b][i];
        }
    }
    fprintf(fp_log, "calculating delta of the %dth layer is completed\n", neural_network -> v_layer -> k);
    print_network(neural_network, fp_log);
    //fprintf(fp_log, "neural_network is at %dth layer.\n", neural_network -> v_layer -> k); // デバグコード

    return neural_network;  // 線形リストの末尾へのポインタを返す
}



// 6. 関数```back_prop```を定義する
//    - 誤差逆伝播を行い```neural_network```を更新する
//    - 引数はネットワーク層構造体の線形リストの末尾要素へのポインタ```neural_network```と、学習率```alpha```と、バッチサイズ```b_size```
//    - まず、第 $D+1$ ニューロン層の```delta```を用いて、第 $D+1$ ネットワーク層の行列成分 $U[i][j]$ を更新する
//       - 「第 $D$ ニューロン層の```o```と第 $D+1$ ニューロン層の```delta```の積」の $0 <= b < b_size$ での平均```del_Uij```を、各i,jの組について計算する
//       - 第 $D+1$ ネットワーク層の行列成分 $U[i][j]$ は $U[i][j]-  alpha* del_Uij$ に更新される
//    - 次に、第 $D$ ニューロン層の、第 $b$ データの第 $i$ ニューロンの```delta```を更新する ($0 <= b < b_size$)
//       - 「第 $D+1$ ニューロン層の第 $l$ ニューロンの```delta```の値と、第 $D+1$ ニューロン層の行列成分 $U[l][i]$ の値の積」の $0<=l<N_{D+1}$ での和に、更新したいニューロンの```delA```を掛けたものを```delta```とする
//    - 以下同様に、ネットワーク層の行列成分の更新と、ニューロン層の各ニューロンの```delta```の更新を繰り返す
//      - 第 $1$ ネットワーク層の行列成分の更新で、全体の更新を終える
//    - 線形リストの先頭のネットワーク層構造体へのポインタを返す

network_1layer *back_prop(FILE *fp_log, network_1layer *neural_network, double alpha, int b_size) {
    fprintf(fp_log, "\n===============<<back_prop>>================\n");

    // 第D+1層の行列成分Uを更新
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
            }
        }
    }
    fprintf(fp_log, "updating U of the %dth layer is completed\n", neural_network -> v_layer -> k);

    // 第D層のdeltaを更新
    for (int b = 0; b < b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            double delta_sum = 0.0;
            for (int l = 0; l < neural_network -> v_dim; l++) {
                // delta_sum += (neural_network -> v_layer -> neuron_2darray[b][l].delta) * (neural_network -> weight[l][i]);
                delta_sum += (neural_network -> v_layer -> neuron_2darray[b][l].delta) * (neural_network -> weight[i][l]);
            }
            (neural_network -> o_layer -> neuron_2darray[b][i].delta) = delta_sum * (neural_network -> o_layer -> neuron_2darray[b][i].delA);
            // 浮動小数点のオーバーフローを防ぐため、0に近い値は0にする
            if (fabs(neural_network -> o_layer -> neuron_2darray[b][i].delta) <= 0.00001) {
                (neural_network -> o_layer -> neuron_2darray[b][i].delta) = 0.0;
            }
        }
    }
    fprintf(fp_log, "updating delta of the %dth layer is completed\n", neural_network -> o_layer -> k);
    print_network(neural_network, fp_log);

    // 第D層から第2層まで、ネットワーク層の行列成分Uとo_layerのdeltaを交互に更新
    while (neural_network -> o_layer -> k > 1) {
        neural_network = neural_network -> prev;
        fprintf(fp_log, "updating U (%dth layer) and delta of the %dth layer\n", neural_network -> v_layer -> k ,neural_network -> o_layer -> k);
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
                    fprintf(fp_log, "U of the %dth layer is too large\n", neural_network -> v_layer -> k);
                }
            }
        }
        fprintf(fp_log, "updating U of the %dth layer is completed\n", neural_network -> v_layer -> k);
        print_network(neural_network, fp_log);
        // printf("updating U of the %dth layer is completed\n", neural_network -> v_layer -> k);
        // ここまでエラーなし

        // // 第kネットワーク層のo_layerのdeltaを更新
        // if (neural_network -> o_layer -> k == 1){
        //     printf("neural_network -> o_dim is %d.\n", neural_network -> o_dim);
        //     printf("neural_network -> v_dim is %d.\n", neural_network -> v_dim);
        //     // weightを表示
        //     for (int i = 0; i < neural_network -> o_dim; i++) {
        //         for (int j = 0; j < neural_network -> v_dim; j++) {
        //             printf("%f ", neural_network -> weight[i][j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("weight is displayed\n");
        // }
        for (int b = 0; b < b_size; b++) {
            for (int i = 0; i < neural_network -> o_dim; i++) {
                double delta_sum = 0.0;
                for (int l = 0; l < neural_network -> v_dim; l++) {
                    // delta_sum += (neural_network -> v_layer -> neuron_2darray[b][l].delta) * (neural_network -> weight[l][i]);
                    delta_sum += (neural_network -> v_layer -> neuron_2darray[b][l].delta) * (neural_network -> weight[i][l]);
                }
                // デバッグ用
                // if (neural_network -> o_layer -> k == 1) {printf("delta_sum: %f\n", delta_sum);}
                (neural_network -> o_layer -> neuron_2darray[b][i].delta) = delta_sum * (neural_network -> o_layer -> neuron_2darray[b][i].delA);
                // 浮動小数点のオーバーフローを防ぐため、0に近い値は0にする
                if (fabs(neural_network -> o_layer -> neuron_2darray[b][i].delta) <= 0.00001) {
                    (neural_network -> o_layer -> neuron_2darray[b][i].delta) = 0.0;
                } else if (fabs(neural_network -> o_layer -> neuron_2darray[b][i].delta) >= 1000000) {
                    fprintf(fp_log, "delta of the %dth layer is too large. Please re-design neural network or learning rate!\n", neural_network -> o_layer -> k);
                    exit(1);
                }
            }
        }
        fprintf(fp_log, "updating delta of the %dth layer is completed\n", neural_network -> o_layer -> k);
        print_network(neural_network, fp_log);
    }
    fprintf(fp_log, "updating U and delta of the 2nd layer is completed\n");
    // fprintf(fp_log, "neural_network is at %dth layer.\n", neural_network -> o_layer -> k); // デバグコード
    neural_network = neural_network -> prev;    // 学習が進まないバグを解消するため追加 2024/03/07

    // 第1ネットワーク層の行列成分Uを更新
    for (int i = 0; i < neural_network -> o_dim; i++) {
        for (int j = 0; j < neural_network -> v_dim; j++) {
            double del_Uij = 0.0;
            for (int b = 0; b < b_size; b++) {
                del_Uij += (neural_network -> v_layer -> neuron_2darray[b][j].delta) * (neural_network -> o_layer -> neuron_2darray[b][i].o);
            }
            neural_network -> weight[i][j] -= alpha * del_Uij / b_size;
            // // 浮動小数点のオーバーフローを防ぐため、0に近い値は0にする
            // if (neural_network -> weight[i][j] <= 0.00001) {
            //     neural_network -> weight[i][j] = 0.0;
            // }
        }
    }
    fprintf(fp_log, "updating U of the 1st layer is completed\n");
    // fprintf(fp_log, "neural_network is at %dth layer.\n", neural_network -> o_layer -> k); // デバグコード
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

// 7. 活性化関数```leakly_relu```を定義する
//   - 引数には、```v```をとる
//   - ```v```が0以上なら```v```を返し、0未満なら```0.01*v```を返す

double leakly_relu(double v) {
    if (v >= 0) {
        // return 0.1 * v;
        return v;
    }
    else {
        return 0.001 * v;
    }
}

double leakly_relu_grad(double v) {
    if (v >= 0) {
        // return 0.1;
        return 1.0;
    }
    else {
        return 0.001;
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