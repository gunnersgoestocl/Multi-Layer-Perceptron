#include "eval_func.h"

// R2値の計算
//    - 引数には、目的変数値の二次元配列```y```と、出力の推定量の二次元配列```o```をとる
//    - 各行ごとに決定係数を計算し、その平均値を返す
//    - 決定係数は、線形回帰ではないので、 で計算する
double calc_r2(double **y, double **o, int b_size, int v_dim) {
    double r2 = 0.0;

    // 今回は出力が1次元の場合のみを想定するため、ループも一重であるし、```v_dim```も使わない
    double *y_sum = (double *)malloc(sizeof(double) * v_dim); // 出力の各次元の値の、バッチデータに対する合計値
    for (int i = 0; i < v_dim; i++) {
        y_sum[i] = 0.0;
    }
    for (int b = 0; b < b_size; b++) {
        for (int i = 0; i < v_dim; i++) {
            y_sum[i] += y[b][i];
        }
    }
    // 出力の各次元の平均値
    double *y_mean = (double *)malloc(sizeof(double) * v_dim);
    for (int i = 0; i < v_dim; i++) {
        y_mean[i] = y_sum[i] / b_size;
    }

    // 各次元の Total Sum of Squares (TSS)の計算
    double *TSS = (double *)malloc(sizeof(double) * v_dim);
    for (int i = 0; i < v_dim; i++) {
        for (int b = 0; b < b_size; b++) {
            TSS[i] += pow(y[b][i] - y_mean[i], 2);
        }
    }

    // 各次元の Residual Sum of Squares (RSS)の計算
    double *RSS = (double *)malloc(sizeof(double) * v_dim);
    for (int i = 0; i < v_dim; i++) {
        for (int b = 0; b < b_size; b++) {
            RSS[i] += pow(y[b][i] - o[b][i], 2);
        }
    }
    
    // 決定係数の計算
    double *r2_each = (double *)malloc(sizeof(double) * v_dim);
    for (int i = 0; i < v_dim; i++) {
        r2_each[i] = 1 - RSS[i] / TSS[i];
    }

    // 決定係数の平均値を計算
    for (int i = 0; i < v_dim; i++) {
        r2 += r2_each[i];
    }
    
    return r2;
}

// 正解率の計算
//    - 引数には、目的変数値の一次元配列(正解クラスのindexが格納)```y```と、出力の推定量の二次元配列```o```をとる
//    - データごとに正解率を計算し、その平均値を返す
double calc_accuracy(double **y, double **o, int b_size, int v_dim) {
    double accuracy = 0.0;

    // 今回は出力が1次元の場合のみを想定するため、ループも一重であるし、```v_dim```も使わない
    int correct = 0;
    for (int b = 0; b < b_size; b++) {
        if (y[b][0] == o[b][0]) {
            correct++;
        }
    }
    accuracy = (double)correct / b_size;
    
    return accuracy;
}