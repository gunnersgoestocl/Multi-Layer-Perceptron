#include "loss_func.h"

loss_func LOSS_FUNC[] = {MSE, BCE, CCE, KLdivergence, };

struct Entry_ {
    int out_activator;
    int loss_func;
    int out_delta;
};

Entry DICT[] = {
    {0, 0, 0},  // MSE & Identity
    {3, 1, 0},  // BCE & Sigmoid
    {10, 2, 0}, // CCE & Softmax
};

size_t dict_size = sizeof(DICT) / sizeof(DICT[0]);

/* loss function (y:estimate, t:target) */
double MSE(double **y, double **t, int data_size, int v_dim, int *v_class) {
    double loss = 0.0;
    for (int i = 0; i < data_size; i++) {
        for (int j = 0; j < v_dim; j++) {
            loss += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]);
        }
    }
    return loss / (2 * data_size);
}

double BCE(double **y, double **t, int data_size, int v_dim, int *v_class) {
    double loss = 0.0;
    for (int i = 0; i < data_size; i++) {
        for (int j = 0; j < v_dim; j++) {
            loss += - t[i][j] * log(y[i][j]) - (1 - t[i][j]) * log(1 - y[i][j]);
        }
    }
    return loss / data_size;
}

double CCE(double **y, double **t, int data_size, int v_dim, int *v_class) {
    double loss = 0.0;
    for (int i = 0; i < data_size; i++) {
        int idx = 0;
        for (int j = 0; j < v_dim; j++) {
            double loss_unit = 0.0;
            for (int k = 0; k < v_class[j]; k++) {
                loss_unit += - t[i][idx] * log(y[i][idx]);
                idx++;
            }
            loss += loss_unit / v_class[j];
        }
    }
    return loss / data_size;
}

double KLdivergence(double **y, double **t, int data_size, int v_dim, int *v_class) {
    double loss = 0.0;
    for (int i = 0; i < data_size; i++) {
        for (int j = 0; j < v_dim; j++) {
            loss += t[i][j] * log(t[i][j] / y[i][j]);
        }
    }
    return loss / data_size;
}

// 2つのキーで値を取得する関数
int get_value(int out_activator, int loss_func) {
    for (size_t i = 0; i < dict_size; i++) {
        if (DICT[i].out_activator == out_activator && DICT[i].loss_func == loss_func) {
            return DICT[i].out_delta;
        }
    }
    fprintf(stderr, "Key pair (%d, %d) not found.\n", out_activator, loss_func);
    return -1;  // エラー値
}

/* output layer delta */
out_delta_func OUT_DELTA_FUNC[] = {error, };

double error(double y, double t) {
    return y - t;
}