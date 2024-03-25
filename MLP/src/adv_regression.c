#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "init.h"
#include "setting.h"
#include "learn.h"
#include "graph.h"
// #include "prop.h"



// main関数
int main(int argc, char **argv) {

    // <<<<<<<<<<<<<<<<<<FILE Pointerの生成>>>>>>>>>>>>>>>>>>>>>>>>>>
    // コマンドラインから分析したいファイル名を読み込む
    char *filename_data = argv[1];
    FILE *fp = fopen(filename_data, "r");
    if (fp == NULL) {
        printf("Cannot open %s\n", filename_data);
        exit(1);
    }

    // コマンドラインから分析の過程の出力先のファイルを読み込む
    char *filename_log = argv[2];
    FILE *fp_log = fopen(filename_log, "w");
    if (fp_log == NULL) {
        printf("Cannot open %s\n", filename_log);
        exit(1);
    }

    char *filename_graph;
    int fprint_graph = 0;
    // コマンドラインから学習における精度の推移を出力するファイルを読み込む
    if (argc == 4) {
        filename_graph = argv[3];
        fprint_graph = 1;
    }

    // <<<<<<<<<<<<<<<<<<<<<<<<変数の取得>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    int *v_col = (int *)malloc(sizeof(int));    // 目的変数の列番号を格納する配列
    int v_dim = 0;  // 目的変数の個数
    int o_dim = 0;  // 説明変数の個数
    int *o_col = (int *)malloc(sizeof(int));    // 説明変数の列番号を格納する配列

    // この上にあった変数設定部分を関数化する
    set_variables(fp, fp_log, &v_dim, v_col, &o_dim, o_col);
    
    printf("Start designing the neural network.\n");
    
    // <<<<<<<<ニューラルネットワークの設計>>>>>>>>>>>>>

    int D;  // ニューラルネットワークの層の数
    double alpha = 0.00001; // 学習率
    int *N = design_network(fp_log, &D, o_dim, v_dim, &alpha);  // 各層のニューロンの数を格納する配列

    // <<<<<<<<<<<<学習の仕方の設定>>>>>>>>>>>>>

    int data_size; // 1行目は変数名なので、データの行数は1つ少ない
    int b_size; // バッチサイズ
    int iter;   // イテレーション数
    int test_size;   // テストデータのサイズ
    int test_iter;   // テストデータのイテレーション数
    int total_epoch; // エポック数
    int cross_val = 0;      // 交差検証を行うかのフラグ

    set_learning(fp, fp_log, &data_size, &b_size, &iter, &test_size, &test_iter, &total_epoch, &cross_val);

    // <<<<<<<<<<<<<<記録の設定>>>>>>>>>>>>>>>

    int program_confirmation;
    int show_values;

    // logファイルに記録する内容を指定する
    printf("You can choose the contents to be recorded in the log file.\n");
    printf("The contents shown below will be necessarily recorded in the log file.\n");
    printf("    1. The objective variable\n");
    printf("    2. The explanatory variables\n");
    printf("    3. The number of neurons in each layer\n");
    printf("    4. The learning rate\n");
    printf("    5. The total number of data\n");
    printf("    6. The batch size\n");
    printf("    7. The number of iterations\n");
    printf("    8. The size of test data\n");
    printf("    9. The total number of epochs\n");
    printf("    10. The coefficient of determination of test data\n");

    printf("Then, you can choose the additional contents to be recorded in the log file.\n");

    printf("Do you want the confirmation sentences that the program is working properly?\n");
    printf("    1:yes   2:no    (input 1 or 2) :\n");
    scanf("%d", &program_confirmation);

    printf("Do you want the values which neurons and weights take in every stage?\n");
    printf("    1:yes   2:no    3:only if iteration ends   (input 1,2 or 3) :\n");
    scanf("%d", &show_values);

    // 学習時の決定係数の推移をグラフで表示する
    FILE *fp_graph;
    if (fprint_graph == 1) {
        fp_graph = popen("gnuplot", "w");
        fprintf(fp_graph, "set terminal png\n");
        fprintf(fp_graph, "set output \"%s.png\"\n", filename_graph);
        fprintf(fp_graph, "set title \"The growth of the coefficient of determination\"\n");
        fprintf(fp_graph, "set xlabel \"epoch\"\n");
        fprintf(fp_graph, "set ylabel \"coefficient of determination\"\n");
        fprintf(fp_graph, "set yrange[-300.0:1.0]\n");
        fprintf(fp_graph, "plot");
        for (int i = 0; i < (iter)+1; i++) {
            fprintf(fp_graph, " '-' with lines title 'Cross Validation #%d',", i+1);
        }
        fprintf(fp_graph, "\n");
    }

    // 現状は、テストデータの読み込みタイミングを、ミニバッチに分割したうち、どのバッチに対して行うかで指定している
    int test_index[b_size*test_iter];    // テストデータの元データにおけるインデックスを格納する配列    今後必要になる可能性あり

    // <<<<<<<<<<<<<<データの読み込みと学習>>>>>>>>>>>>>>>
    // 設計したモデルについて、データを与えて、交差検証法により、学習とテストを両方行う関数
    // 各検証において、訓練時の学習決定係数の推移がグラフに出力され、最終的なテストデータにおける決定係数が列挙される
    CRVL(total_epoch, iter, b_size, test_iter, D, N, o_dim, o_col, v_dim, v_col, alpha, leakly_relu, leakly_relu_grad, program_confirmation, show_values, fprint_graph, fp, fp_log, fp_graph);

    if (fprint_graph == 1) {
        fprintf(fp_graph, "quit\n");
        pclose(fp_graph);
    }

    return 0;
}
