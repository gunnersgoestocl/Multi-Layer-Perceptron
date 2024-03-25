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

double calc_r2(double **y, double **o, int b_size, int v_dim);
double test(double **x_test, double **y_test, int b_size, int test_iter, int v_dim, network_1layer *neural_network, double (*activator)(double), double (*activator_grad)(double), int program_confirmation, int show_values, FILE *fp_log, FILE *fp_graph);

void load_learn(int total_epoch, int iter, int CRVL, int b_size, int test_iter, double **x_test, double **y_test, int o_dim, int *o_col, int v_dim, int *v_col, network_1layer *neural_network, double alpha, double (*activator)(double), double (*activator_grad)(double), int program_confirmation, int show_values, int fprint_graph, FILE *fp, FILE *fp_log, FILE *fp_graph);
void CRVL(int total_epoch, int iter, int b_size, int test_iter, int D, int *N, int o_dim, int *o_col, int v_dim, int *v_col, double alpha, double (*activator)(double), double (*activator_grad)(double), int program_confirmation, int show_values, int fprint_graph, FILE *fp, FILE *fp_log, FILE *fp_graph);


#endif