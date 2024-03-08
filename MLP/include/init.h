#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef _INIT_H_
#define _INIT_H_

// 1. 構造体```neuron```を定義する
//   - ニューロンへの入力値```v```をメンバに持つ
//   - ニューロンからの出力値```o```をメンバに持つ
//   - 損失関数のニューロンへの入力値に対する勾配```delta```をメンバに持つ
//   - 活性化関数のニューロンへの入力値に対する勾配```delA```をメンバに持つ
typedef struct neuron_ neuron;

// 2. ニューロンの層を表す構造体の二次元配列```neuron_2darray```をメンバに持つ構造体```neuron_layer```を定義する
//   - 各層のニューロンを表す構造体定義```neuron```を参照する
//   - 層番号```k```をメンバに持つ
//   - その層のニューロンの個数 $N_k$ をメンバに持つ
//   - ニューロン層を表す構造体の初期化関数```init_neuron_layer```を定義し、ニューロンの層を表す構造体の二次元配列```neuron_2darray```を含めて構造体を初期化する
//     - 引数には、層番号```k```と各中間層のニューロンの数```N_k```とバッチサイズ```b_size```をとる

typedef struct neuron_layer_ neuron_layer;

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

neuron_layer *init_neuron_layer(int k, int N_k, int b_size);
network_1layer *rewind_network(network_1layer *neural_network);
network_1layer *push_network_back(neuron_layer *o_layer, neuron_layer *v_layer, network_1layer *neural_network);

void free_neuron_layer(neuron_layer *layer);
void free_network_1layer(network_1layer *network);
void free_network(network_1layer *neural_network);

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

network_1layer *forward_prop(FILE *fp_log, double **x, double **y, network_1layer *neural_network, double (*activator)(double), double (*activator_grad)(double), int program_confirmation, int show_values);

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

network_1layer *back_prop(FILE *fp_log, network_1layer *neural_network, double alpha, int b_size, int program_confirmation, int show_values);

//ニューラルネットワークを表す構造体へのポインタ、ネットワークの格納する値をグラフに可視化して出力するファイルのポインタを引数に取る
void print_network(network_1layer *neural_network, FILE *fp_log);

double leakly_relu(double v);
double leakly_relu_grad(double v);

double **estimate(int b_size, int v_dim, network_1layer *neural_network);
void estimate_sub (int b_size, int iter_num, int v_dim, network_1layer *neural_network, double **y_hat);

#endif