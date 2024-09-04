#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef _ACTIVATOR_H_
#define _ACTIVATOR_H_

/* 構造体に依存しない活性化関数 */
double linear(double v);
double linear_grad(double v);
double relu(double v);
double relu_grad(double v);
double leakly_relu(double v);
double leakly_relu_grad(double v);
double sigmoid(double v);
double sigmoid_grad(double o);
double tanh(double v);
double tanh_grad(double o);

/* 活性化関数の関数ポインタの配列を定義 */
typedef double (*activators)(double);

/* 活性化関数の勾配の関数ポインタの配列を定義 */
typedef double (*activators_grad)(double);

extern activators ACTIVATORS[];
extern activators_grad ACTIVATORS_GRAD[];

// softmaxはニューロン層全体で計算するため、init.hに記述
// void softmax(int start_idx, int v_class, neuron *neuron_1darray);

#endif