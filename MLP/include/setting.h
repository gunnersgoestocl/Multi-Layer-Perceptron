#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef _SETTING_H_
#define _SETTING_H_

void set_variables(FILE *fp, FILE *fp_log, int *v_dim, int *v_col, int *o_dim, int *o_col);
int *design_network(FILE *fp_log, int *D, int o_dim, int v_dim, double *alpha);

void set_learning(FILE *fp, FILE *fp_log, int *data_size, int *b_size, int *iter, int *test_size, int *test_iter, int *total_epoch);




#endif