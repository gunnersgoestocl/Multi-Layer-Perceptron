#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef _EVAL_FUNC_H_
#define _EVAL_FUNC_H_

double calc_r2(double **y, double **o, int b_size, int v_dim);
double calc_accuracy(double **y, double **o, int b_size, int v_dim);

#endif