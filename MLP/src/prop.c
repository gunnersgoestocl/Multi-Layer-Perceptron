#include "prop.h"
#include "init.h"

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
