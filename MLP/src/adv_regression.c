#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "init.h"
#include "setting.h"
#include "learn.h"


// main関数
int main(int argc, char **argv) {

    // <<<<<<<<<<<<<<<<<<FILE Pointerの生成>>>>>>>>>>>>>>>>>>>>>>>>>>   Generation of FILE Pointer
    // コマンドラインから分析したいファイル名を読み込む     Read the file name you want to analyze from the command line
    char *filename_data = argv[1];
    FILE *fp = fopen(filename_data, "r");
    if (fp == NULL) {
        printf("Cannot open %s\n", filename_data);
        exit(1);
    }

    // コマンドラインから分析の過程の出力先のファイルを読み込む     Read the file to output the process of analysis from the command line
    char *filename_log = argv[2];
    FILE *fp_log = fopen(filename_log, "w");
    if (fp_log == NULL) {
        printf("Cannot open %s\n", filename_log);
        exit(1);
    }

    char *filename_graph;
    int fprint_graph = 0;
    // コマンドラインから学習における精度の推移を出力するファイルを読み込む  Read the file to output the transition of accuracy in learning from the command line
    if (argc == 4) {
        filename_graph = argv[3];
        fprint_graph = 1;
    }

    // <<<<<<<<<<<<<<<<<<<<<<<<変数の指定>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Acquisition of variables
    int *v_col = (int *)malloc(sizeof(int));    // 目的変数の列番号を格納する配列   Array to store the column number of the objective variable
    int v_dim = 0;  // 目的変数の個数   Number of objective variables
    int o_dim = 0;  // 説明変数の個数   Number of explanatory variables
    int *o_col = (int *)malloc(sizeof(int));    // 説明変数の列番号を格納する配列   Array to store the column number of the explanatory variable

    // 上記の変数を設定する Set the above variables
    set_variables(fp, fp_log, &v_dim, v_col, &o_dim, o_col);
    
    // <<<<<<<<ニューラルネットワークの設計>>>>>>>>>>>>>    Designing a neural network

    int D;  // ニューラルネットワークの層の数 Number of layers in the neural network
    double alpha = 0.00001; // 学習率   Learning rate
    int *N = design_network(fp_log, &D, o_dim, v_dim, &alpha);  // 各層のニューロンの数を格納する配列   Array to store the number of neurons in each layer

    // <<<<<<<<<<<<学習の仕方の設定>>>>>>>>>>>>>    Setting how to learn

    int data_size; // 1行目は変数名なので、データの行数は1つ少ない
    int b_size; // バッチサイズ Number of data in a batch
    int iter;   // イテレーション数 Number of iterations
    int test_size;   // テストデータのサイズ Number of test data
    int test_iter;   // テストデータのイテレーション数 Number of iterations for test data
    int total_epoch; // エポック数 Number of epochs
    int cross_val = 0;      // 交差検証を行うかのフラグ Flag to perform cross validation

    set_learning(fp, fp_log, &data_size, &b_size, &iter, &test_size, &test_iter, &total_epoch, &cross_val);

    // <<<<<<<<<<<<<<記録の設定>>>>>>>>>>>>>>>  Setting of recording

    int program_confirmation;   // プログラムの動作確認文を記録するかのフラグ Flag to record the program confirmation sentence
    int show_values;    // 各ステージでのニューロンの値を記録するかのフラグ Flag to record the value of neurons at each stage

    // logファイルに記録する内容を指定する      Specify the contents to be recorded in the log file
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

    // 学習時の決定係数の推移をグラフで表示する     Display the transition of the coefficient of determination during learning in a graph
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
        if (cross_val == 1) {
            for (int i = 0; i < (iter)+1; i++) {
                fprintf(fp_graph, " '-' with lines title 'Cross Validation #%d',", i+1);
            }
        } else {
            fprintf(fp_graph, " '-' with lines title 'Learning'");
        }
        fprintf(fp_graph, "\n");
    }

    // // 現状は、テストデータの読み込みタイミングを、ミニバッチに分割したうち、どのバッチに対して行うかで指定しているため不要
    // int test_index[b_size*test_iter];    // テストデータの元データにおけるインデックスを格納する配列     Array to store the index in the original data of the test data

    // <<<<<<<<<<<<<<データの読み込みと学習>>>>>>>>>>>>>>>  Reading data and learning
    
    // 各検証において、訓練時の学習決定係数の推移がグラフに出力され、最終的なテストデータにおける決定係数が列挙される   In each validation, the transition of the training coefficient of determination is output to a graph, and the coefficient of determination in the final test data is enumerated
    if (cross_val == 1) {   // 交差検証を行う
        // 設計したモデルについて、データを与えて、交差検証法により、学習とテストを両方行う関数     A function that provides data for the designed model and performs both learning and testing by cross-validation
        CRVL(total_epoch, iter, b_size, test_iter, D, N, o_dim, o_col, v_dim, v_col, alpha, leakly_relu, leakly_relu_grad, program_confirmation, show_values, fprint_graph, fp, fp_log, fp_graph);
    } else {
        // テストデータの位置を、0以上、iter以下の整数からランダムに選ぶ
        // 乱数の種を時間で設定
        srand((unsigned)time(NULL));
        int CRVL = rand() % (iter+1);
        learn_set(total_epoch, iter, CRVL, b_size, test_iter, D, N, o_dim, o_col, v_dim, v_col, alpha, leakly_relu, leakly_relu_grad, program_confirmation, show_values, fprint_graph, fp, fp_log, fp_graph);
    }

    if (fprint_graph == 1) {
        fprintf(fp_graph, "quit\n");
        pclose(fp_graph);
    }
    printf("Program is completed.\n");

    return 0;
}
