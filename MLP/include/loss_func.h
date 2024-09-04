#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef _LOSS_FUNC_H_
#define _LOSS_FUNC_H_

/* 損失関数の関数ポインタの配列を定義 (実際に損失関数を計算する必要はないため使わない) */
typedef double (*loss_func)(double **, double **, int, int, int *);

double MSE (double **y, double **t, int data_size, int v_dim, int *v_class);
double BCE (double **y, double **t, int data_size, int v_dim, int *v_class);
double CCE (double **y, double **t, int data_size, int v_dim, int *v_class);
double KLdivergence (double **y, double **t, int data_size, int v_dim, int *v_class);

extern loss_func LOSS_FUNC[];

/* 出力層のactivatorと損失関数のidxをキーとした、出力層のdeltaを計算する関数の連想配列 */
typedef struct Entry_ Entry;

int get_value(int out_activator, int loss_func);

/* 出力層のdelta (∂L/∂v) を計算する関数の関数ポインタの配列 */
typedef double (*out_delta_func)(double, double);

double error(double y, double t);

extern out_delta_func OUT_DELTA_FUNC[];

#endif