#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// 1. 構造体```neuron```を定義する
//   - ニューロンへの入力値```v```をメンバに持つ
//   - ニューロンからの出力値```o```をメンバに持つ
//   - 損失関数のニューロンへの入力値に対する勾配```delta```をメンバに持つ
//   - 活性化関数のニューロンへの入力値に対する勾配```delA```をメンバに持つ

typedef struct {
  double v;
  double o;
  double delta;
  double delA;
} neuron;

// 2. ニューロンの層を表す構造体の二次元配列```neuron_2darray```をメンバに持つ構造体```neuron_layer```を定義する
//   - 各層のニューロンを表す構造体定義```neuron```を参照する
//   - 層番号```k```をメンバに持つ
//   - その層のニューロンの個数 $N_k$ をメンバに持つ
//   - ニューロン層を表す構造体の初期化関数```init_neuron_layer```を定義し、ニューロンの層を表す構造体の二次元配列```neuron_2darray```を含めて構造体を初期化する
//     - 引数には、層番号```k```と各中間層のニューロンの数```N_k```とバッチサイズ```b_size```をとる

typedef struct {
  neuron **neuron_2darray;
  int k;
  int N_k;
  int b_size;
} neuron_layer;

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

// 3. 構造体```network_1layer```を定義する
//   - 各層間の重みを表す行列へのポインタをメンバに持つ
//     - この行列は、(ネットワークに入力を与える層のニューロン数) $N_k \times$ (ネットワークから値を受け取る層のニューロン数) $N_{k+1}$ の行列である
//   - ネットワークに入力を与える「ニューロン層」の```neuron_layer```構造体へのポインタ```o_layer```をメンバに持つ
//   - ネットワークから値を受け取る「ニューロン層」の```neuron_layer```構造体へのポインタ```v_layer```をメンバに持つ
//   - 入力層(ネットワークに値を入力する)のニューロン数```o_dim```をメンバに持つ(```o_layer -> N_k```)
//   - 出力層(ネットワークから値を受け取る層)のニューロン数```v_dim```をメンバに持つ(```v_layer -> N_k```)
//   - 手前のネットワーク層構造体へのポインタ```prev```をメンバに持つ
//   - 次のネットワーク層構造体へのポインタ```next```をメンバに持つ

typedef struct network_1layer_ network_1layer;
struct network_1layer_ {
  double **weight;
  neuron_layer *o_layer;
  neuron_layer *v_layer;
  int o_dim;
  int v_dim;
  network_1layer *prev;
  network_1layer *next;
};

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

network_1layer *forward_prop(double **x, double **y, network_1layer *neural_network, double (*activator)(double), double (*activator_grad)(double)) {

    printf("forward_prop\n");
    // 第0層の入力値vをxに設定
    for (int b = 0; b < (neural_network -> o_layer) -> b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            neural_network -> o_layer -> neuron_2darray[b][i].v = x[b][i];
        }
    }
    printf("setting v of the 0th layer is completed\n");
    // 第0層の出力値oを更新
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> o_dim; i++) {
            neural_network -> o_layer -> neuron_2darray[b][i].o = neural_network -> o_layer -> neuron_2darray[b][i].v;
        }
    }
    printf("updating o of the 0th layer is completed\n");
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
        // 第k層のニューロンの活性化関数の勾配delAを計算
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].delA = (*activator_grad)(neural_network -> v_layer -> neuron_2darray[b][i].v);
            }
        }
        // 第k層のニューロンへの入力値vを活性化関数にかけて出力値oを計算
        for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
            for (int i = 0; i < neural_network -> v_dim; i++) {
                neural_network -> v_layer -> neuron_2darray[b][i].o = (*activator)(neural_network -> v_layer -> neuron_2darray[b][i].v);
            }
        }
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
    // 第D+1層のニューロンの出力値oを計算
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> v_dim; i++) {
            neural_network -> v_layer -> neuron_2darray[b][i].o = neural_network -> v_layer -> neuron_2darray[b][i].v;
        }
    }
    // 第D+1層のニューロンのdeltaを計算
    for (int b = 0; b < neural_network -> o_layer -> b_size; b++) {
        for (int i = 0; i < neural_network -> v_dim; i++) {
            neural_network -> v_layer -> neuron_2darray[b][i].delta = neural_network -> v_layer -> neuron_2darray[b][i].o - y[b][i];
        }
    }

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

network_1layer *back_prop(network_1layer *neural_network, double alpha, int b_size) {
    printf("back_prop\n");

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
    printf("updating U of the %dth layer is completed\n", neural_network -> v_layer -> k);

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
    printf("updating delta of the %dth layer is completed\n", neural_network -> o_layer -> k);

    // 第D層から第2層まで、ネットワーク層の行列成分Uとo_layerのdeltaを交互に更新
    while (neural_network -> o_layer -> k > 1) {
        neural_network = neural_network -> prev;
        printf("updating U (%dth layer) and delta of the %dth layer\n", neural_network -> v_layer -> k ,neural_network -> o_layer -> k);
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
                    printf("U of the %dth layer is too large\n", neural_network -> v_layer -> k);
                }
            }
        }
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
                    printf("delta of the %dth layer is too large. Please re-design neural network!\n", neural_network -> o_layer -> k);
                    exit(1);
                }
            }
        }
    }
    printf("updating U and delta of the 2nd layer is completed\n");

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
    printf("updating U of the 1st layer is completed\n");

    return neural_network;
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

// 8. 決定係数を計算する関数```calc_r2```を定義する
//    - 引数には、目的変数値の二次元配列```y```と、出力の推定量の二次元配列```o```をとる
//    - 各行ごとに決定係数を計算し、その平均値を返す
//    - 決定係数は、線形回帰ではないので、 で計算する

double calc_r2(double **y, double **o, int b_size, int v_dim) {
    double r2 = 0.0;

    // 今回は出力が1次元の場合のみを想定するため、ループも一重であるし、```v_dim```も使わない
    double y_sum = 0.0;
    // double o_sum = 0.0;
    double y_mean = 0.0;
    // double o_mean = 0.0;
    double y_var = 0.0;
    // double o_var = 0.0;
    // double cov = 0.0;
    double sq_hensa = 0.0;
    for (int b = 0; b < b_size; b++) {
        y_sum += y[b][0];   // 出力が1次元でない場合は要変更
        // o_sum += o[b][0];
    }
    // o_mean = o_sum / b_size;
    y_mean = y_sum / b_size;
    for (int b = 0; b < b_size; b++) {
        y_var += pow(y[b][0] - y_mean, 2);
        // o_var += pow(o[b][0] - o_mean, 2);
        sq_hensa += pow(y[b][0] - o[b][0], 2);
    }
    r2 = 1 - sq_hensa / y_var;
    
    return r2;
}


// main関数
int main(int argc, char **argv) {

    // 学習率
    double alpha = 0.00001;

    // コマンドラインから分析したいファイル名を読み込む
    char *filename = argv[1];
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Cannot open %s\n", filename);
        exit(1);
    }

    // ファイルの1行目を';'で区切って、変数名を読み込み表示する
    char buf[256];
    fgets(buf, sizeof(buf), fp);
    char *token = strtok(buf, ";");
    while (token != NULL) {
        printf("%s\n", token);
        token = strtok(NULL, ";\n");
    }

    // 目的変数名の取得
    // 目的変数の変数名をコマンドラインに書き込むよう指示する
    printf("Input the name of the objective variable: ");
    char objective[256];
    scanf("%[^\n]%*1[\n]", objective);
    printf("You input %s.\n", objective);
    // 目的変数の変数名が含まれる列番号を調べる
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);
    token = strtok(buf, ";\n");
    const int v_dim = 1;  // 目的変数の個数(変数にする実装には今回はしない)
    int v_col = -1;   // 目的変数の列番号
    int col = 0;    
    while (token != NULL) {
        // printf("comparing with %s: %d\n", token, strcmp(token, objective));
        if (strcmp(token, objective) == 0) {
            v_col = col;
            // printf("register success\n");
            break;
        }
        token = strtok(NULL, ";\n");
        col++;
    }
    // 目的変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
    while (v_col == -1) {
        printf("Cannot find the objective variable. Please input the correct name of the objective variable.\n");
        char objective[256];
        scanf("%[^\n]%*1[\n]", objective);
        // 目的変数の変数名が含まれる列番号を調べる
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;    
        while (token != NULL) {
            if (strcmp(token, objective) == 0) {
                v_col = col;
                break;
            }
            token = strtok(NULL, ";\n");
            col++;
        }
    }

    // 説明変数名の取得
    // 説明変数の変数名を一つずつコマンドラインに書き込むよう指示する
    // ALLと書いてEnterを押すと、目的変数以外全ての列を説明変数として処理する
    // 何も書かずにEnterを押すと、説明変数の入力を終了する     
    printf("Input the name of the explanatory variables (ALL to select all or END to quit selecting): ");
    char explanatory[256];
    scanf("%[^\n]%*1[\n]", explanatory);
    printf("You input %s.\n", explanatory);
    int o_dim = 0;  // 説明変数の個数
    int *o_col = (int *)malloc(sizeof(int));    // 説明変数の列番号を格納する配列
    col = 0;    // 列番号のカーソル
    while (strcmp(explanatory, "END") != 0 || o_dim == 0) {
        // 説明変数の列番号を調べる
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        // col = 0;

        while (token != NULL) {
            // printf("comparing with %s: %d\n", token, strcmp(token, explanatory));

            // ALLと書いた場合(目的変数以外全ての列を説明変数として処理する)
            if (strcmp(explanatory, "ALL") == 0) {
                if (strcmp(token, objective) != 0) {
                    o_col = (int *)realloc(o_col, sizeof(int) * (o_dim + 1));   // 説明変数の列番号を格納する配列のサイズを1増やす
                    o_col[o_dim] = col;
                    // col++;
                    o_dim++;
                }
                // printf("You input ALL.\n");
                // token = strtok(NULL, ";\n");
            }

            // ENDを入力した場合(説明変数の個数が0の場合を除いて、説明変数の入力を終了する)
            else if (strcmp(explanatory, "END") == 0) {
                // 説明変数の個数が0の場合、説明変数を少なくとも一つ入力するよう指示する
                if (o_dim == 0) {
                    printf("Please input at least one explanatory variable.\n");
                    break;
                }
                break;
            }

            // 何かしらの変数名を書いた場合
            else if (strcmp(token, explanatory) == 0) {
                // 列番号に重複がある場合は重複していると表示する
                printf("v_col: %d, token: %s\n", v_col, token);
                int flag = 0;
                for (int i = 0; i < o_dim; i++) {
                    // 目的変数と説明変数の列番号が重複している場合は、重複していると表示する
                    if (col == v_col) {
                        printf("The column number %d is already registered as objective variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    } else if (o_col[i] == col) {
                        printf("The column number %d is already registered as explanatory variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    }
                    
                }
                // 説明変数の列番号を格納する配列に、説明変数の列番号を追加する
                if (flag == 0) {
                    // 配列サイズを1増やす
                    if (o_dim != 0) {
                        o_col = (int *)realloc(o_col, sizeof(int) * (o_dim + 1));
                    }
                    o_col[o_dim] = col;
                    o_dim++;
                    printf("%s is successfully registered.\n", explanatory);
                    col = 0;    // カーソルを先頭に戻す
                    printf("o_dim: %d, o_col:", o_dim);
                    for (int i = 0; i < o_dim; i++) {
                        printf("%d ", o_col[i]);
                    }
                    printf("\n");
                }
                break;
            }
            token = strtok(NULL, ";\n");
            col++;
        }   // while (token != NULL) の終わり(変数名の探索の終わり)

        // 説明変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
        if (o_dim == 0) {
            printf("Cannot find the explanatory variable. Please input the correct name of the explanatory variable.\n");
            // break;
        }
    
        // 説明変数の列番号を昇順にソートする
        for (int i = 0; i < o_dim - 1; i++) {
            for (int j = i + 1; j < o_dim; j++) {
                if (o_col[i] > o_col[j]) {
                    int tmp = o_col[i];
                    o_col[i] = o_col[j];
                    o_col[j] = tmp;
                }
            }
        }

        // ALL or 説明変数が1つ以上ある中でのENTER の場合、説明変数の入力を終える
        if ((strcmp(explanatory, "END") == 0 && o_dim != 0)|| strcmp(explanatory, "ALL") == 0) {
            break;
        }    

        // それ以外の場合は、次の説明変数を入力する
        printf("Input the name of the explanatory variables (ALL to select all or END to quit selecting): ");
        scanf("%[^\n]%*1[\n]", explanatory);   
        printf("You input %s.\n", explanatory);
    }   // while (strcmp(explanatory, "") != 0) の終わり(説明変数の入力の終わり)

    // 目的変数と説明変数の列番号を表示し、ニューラルネットワークの設計を開始すると宣言する
    printf("The column number of the objective variable is %d.\n", v_col);
    printf("The column number of the explanatory variables are ");
    for (int i = 0; i < o_dim; i++) {
        printf("%d ", o_col[i]);
    }
    printf(".\n");
    printf("Start designing the neural network.\n");

    // ニューラルネットワークの設計
    // ニューラルネットワークの層の数、各層のニューロンの個数をスペース区切りでコマンドラインから読み込む
    printf("Input the number of layers and the number of neurons in each layer: ");
    int D;  // ニューラルネットワークの層の数
    scanf("%d", &D);
    int *N = (int *)malloc(sizeof(int) * D);    // 各層のニューロンの個数を格納する配列
    for (int i = 0; i < D; i++) {
        scanf("%d", &N[i]);
    }

    // ニューラルネットワークの層の数と各層のニューロンの個数を表示する
    printf("The number of  neuron layers is %d.\n", D+2);
    printf("The number of neurons in each layer are \n");
    printf("    input layer: %7d\n", o_dim);
    for (int i = 0; i < D; i++) {
        printf("    %2d middle layer: %3d\n", i+1, N[i]);
    }
    printf("    output layer: %7d\n", v_dim);   // 今回は出力層のニューロンは1つとあらかじめて決めてしまう
    printf("Now Loading Data File...\n");

    // データサイズの取得
    // データの行数を数える
    int data_size = -1; // 1行目は変数名なので、データの行数は1つ少ない
    while (fgets(buf, sizeof(buf), fp) != NULL) {
        data_size++;
    }
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);    // この時点でfpは2行目を指している(データの1行目を読み飛ばした)
    printf("The number of data is %d.\n", data_size);

    // バッチサイズとイテレーション数をコマンドラインから読み込む
    printf("Input the batch size and the number of iterations: ");
    int b_size; // バッチサイズ
    int iter;   // イテレーション数
    scanf("%d %d", &b_size, &iter);
    printf("The batch size is %d.\n", b_size);
    printf("The number of iterations is %d.\n", iter);
    int test_size = data_size - b_size * iter;   // テストデータのサイズ
    int test_iter = (int) (test_size / b_size);  // テストデータのイテレーション数
    // int test_iter = 1;
    printf("So, the size of test data is %d.\n", b_size * test_iter);

    double test_scores[iter+1]; // テストデータのスコアを格納する配列

    // 交差検証開始
    for (int epoch = 0; epoch < iter+1; epoch++) {  
    
    // ニューラルネットワークを初期化する
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
    printf("The neural network has been initialized.\n");
    // ここまでエラーなし 12/30 22:25

    // テストデータの読み込み
    // テストデータの説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
    double **x_test = (double **)malloc(sizeof(double *) * b_size * test_iter);  // test_sizeにするとエラーが出る(当たり前)
    double **y_test = (double **)malloc(sizeof(double *) * b_size * test_iter);
    for (int b = 0; b < b_size * test_iter; b++) {
        x_test[b] = (double *)malloc(sizeof(double) * o_dim);
        y_test[b] = (double *)malloc(sizeof(double) * v_dim);
    }
    int test_flag = 0;  // テストデータの読み込みが完了したら1にする
    printf("The test data has been initialized.\n");

    // データを読み込んでニューラルネットワークを学習させる
    for (int iter_num = 0; iter_num<iter; iter_num++) {

        // テストデータの読み込み
        if (iter_num == epoch && test_flag == 0) {
            // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
            for (int b = 0; b < b_size*test_iter; b++) {
                fgets(buf, sizeof(buf), fp);
                token = strtok(buf, ";\n");
                col = 0;
                while (token != NULL) {
                    for (int i = 0; i < o_dim; i++) {
                        if (col == o_col[i]) {
                            x_test[b][i] = atof(token);
                        }
                    }
                    if (col == v_col) {
                        y_test[b][0] = atof(token);
                    }
                    token = strtok(NULL, ";\n");
                    col++;
                }
            }   // 1バッチのデータ読み込み完了
            printf("The test data has been loaded.\n");
            test_flag = 1;
            iter_num--;
        }

        // 訓練データの読み込み
        else {
        // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
        double **x = (double **)malloc(sizeof(double *) * b_size);
        double **y = (double **)malloc(sizeof(double *) * b_size);
        for (int b = 0; b < b_size; b++) {
            x[b] = (double *)malloc(sizeof(double) * o_dim);
            y[b] = (double *)malloc(sizeof(double) * v_dim);
        }
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
                if (col == v_col) {
                    y[b][0] = atof(token);
                }
                token = strtok(NULL, ";\n");
                col++;
            }
        }   // 1バッチのデータ読み込み完了
        printf("The data of iteration %d has been loaded.\n", iter_num+1);
        // // xの値を表示する
        // for (int b = 0; b < b_size; b++) {
        //     for (int i = 0; i < o_dim; i++) {
        //         printf("%f ", x[b][i]);
        //     }
        //     printf("\n");
        // }
        // ここまでエラーなし 12/30 22:27

        // 順伝播を行う
        neural_network = forward_prop(x, y, neural_network, leakly_relu, leakly_relu_grad);
        printf("The forward propagation of iteration %d has been completed.\n", iter_num+1);
        printf("neural_network is at %dth layer.\n", neural_network -> o_layer -> k);

        // 出力の推定量を二次元配列に格納する
        double **y_hat = (double **)malloc(sizeof(double *) * b_size);
        for (int b = 0; b < b_size; b++) {
            y_hat[b] = (double *)malloc(sizeof(double) * v_dim);
            for (int i = 0; i < v_dim; i++) {
                y_hat[b][i] = neural_network -> v_layer -> neuron_2darray[b][i].o;
            }
        }
        printf("The estimation of iteration %d has been completed. Deviation is shown below.\n", iter_num+1);
        // y_hatの値を表示する
        for (int b = 0; b < b_size; b++) {
            for (int i = 0; i < v_dim; i++) {
                printf("%f ", y_hat[b][i] - y[b][i]);
            }
            // printf("\n");
        }
        printf("\n");
        // ここまでエラーなし 12/30 22:53

        // 誤差逆伝播を行いニューラルネットワークを更新する
        neural_network = back_prop(neural_network, alpha, b_size);
        printf("The back propagation of iteration %d has been completed.\n", iter_num+1);

        // 1イテレーションごとの決定係数を計算し表示する
        double r2 = calc_r2(y, y_hat, b_size, v_dim);
        printf("The coefficient of determination of iteration %d is %f.\n", iter_num+1, r2);
        // neural_network = rewind_network(neural_network);   // ポインタを末尾から先頭に戻す
        }   // else(訓練データを読んだ場合)の終わり
    }

    // // テストデータの読み込み
    // // テストデータの説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
    // double **x_test = (double **)malloc(sizeof(double *) * b_size);  // test_sizeにするとエラーが出る(当たり前)
    // double **y_test = (double **)malloc(sizeof(double *) * b_size);
    // for (int b = 0; b < b_size; b++) {
    //     x_test[b] = (double *)malloc(sizeof(double) * o_dim);
    //     y_test[b] = (double *)malloc(sizeof(double) * v_dim);
    // }
    if (test_flag == 0) {
        // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
        for (int b = 0; b < b_size*test_iter; b++) {
            fgets(buf, sizeof(buf), fp);
            token = strtok(buf, ";\n");
            col = 0;
            while (token != NULL) {
                for (int i = 0; i < o_dim; i++) {
                    if (col == o_col[i]) {
                        x_test[b][i] = atof(token);
                    }
                }
                if (col == v_col) {
                    y_test[b][0] = atof(token);
                }
                token = strtok(NULL, ";\n");
                col++;
            }
        }   // 1バッチのデータ読み込み完了
        printf("The test data has been loaded.\n");
    }
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);    // この時点でfpは2行目を指している(データの1行目を読み飛ばした)

    // // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
    // for (int b = 0; b < b_size; b++) {
    //     fgets(buf, sizeof(buf), fp);
    //     token = strtok(buf, ";\n");
    //     col = 0;
    //     while (token != NULL) {
    //         for (int i = 0; i < o_dim; i++) {
    //             if (col == o_col[i]) {
    //                 x_test[b][i] = atof(token);
    //             }
    //         }
    //         if (col == v_col) {
    //             y_test[b][0] = atof(token);
    //         }
    //         token = strtok(NULL, ";\n");
    //         col++;
    //     }
    // }   // テストデータの読み込み完了

    // テストデータを当てはめた時の決定係数を計算し表示する
    // printf("The test data has been loaded.\n");
    printf("Now calculating the coefficient of determination of test data...\n");
    neural_network = rewind_network(neural_network);   // ポインタを末尾から先頭に戻す
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    double **y_hat = (double **)malloc(sizeof(double *) * b_size * test_iter);
    for (int b = 0; b < b_size * test_iter; b++) {
        y_hat[b] = (double *)malloc(sizeof(double) * v_dim);
    }
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    for (int iter_num = 0; iter_num < test_iter; iter_num++) {
        neural_network = forward_prop(&x_test[iter_num*b_size], &y_test[iter_num*b_size], neural_network, leakly_relu, leakly_relu_grad);
        for (int b = 0; b < b_size; b++) {
            for (int i = 0; i < v_dim; i++) {
                y_hat[iter_num*b_size+b][i] = neural_network -> v_layer -> neuron_2darray[b][i].o;
            }
        }
    }

    // neural_network = forward_prop(x_test, y_test, neural_network, leakly_relu, leakly_relu_grad);
    
    // for (int b = 0; b < b_size; b++) {
    //     y_hat[b] = (double *)malloc(sizeof(double) * v_dim);
    //     for (int i = 0; i < v_dim; i++) {
    //         y_hat[b][i] = neural_network -> v_layer -> neuron_2darray[b][i].o;
    //     }
    // }

    for (int b = 0; b < b_size*test_iter; b++) {
        for (int i = 0; i < v_dim; i++) {
            printf("%f ", y_hat[b][i]);
        }
        // printf("\n");
    }
    printf("\n");

    double r2 = calc_r2(y_test, y_hat, b_size*test_iter, v_dim);
    printf("The coefficient of determination of test data is %f. (layer: %d)\n", r2, neural_network -> v_layer -> k);
    // テストデータの決定係数を配列に格納する
    test_scores[epoch] = r2;

    // メモリの解放
    rewind_network(neural_network); // ポインタを末尾から先頭に戻す
    free_network(neural_network);

    } // 交差検証終了

    // 交差検証の結果を表示する
    printf("--------Review of the coefficient of determination of test data:------------\n");
    double test_score_sum = 0.0;
    for (int epoch = 0; epoch < iter+1; epoch++) {
        printf("The coefficient of determination of test data in epoch %d is %f.\n", epoch+1, test_scores[epoch]);
        test_score_sum += test_scores[epoch];
    }
    printf("The average of the coefficient of determination of test data is %f.\n", test_score_sum / (iter+1));

    return 0;
}
