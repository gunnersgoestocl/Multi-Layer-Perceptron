#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "init.h"

#ifndef _LEARN_H_
#define _LEARN_H_

network_1layer  *init_network(int o_dim, int v_dim, int b_size, int D, int *N, FILE *fp_log, int program_confirmation);
void load_testData(FILE *fp, int *test_flag, int b_size, int test_iter, double **x_test, double **y_test, int o_dim, int *o_col, int v_dim, int *v_col);
void load_Data(FILE *fp, int b_size, double **x, double **y, int o_dim, int *o_col, int v_dim, int *v_col);


#endif