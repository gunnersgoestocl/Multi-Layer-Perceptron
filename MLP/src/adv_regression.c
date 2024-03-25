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

    // int program_confirmation;
    // int show_values;

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
    int program_confirmation;
    scanf("%d", &program_confirmation);

    printf("Do you want the values which neurons and weights take in every stage?\n");
    printf("    1:yes   2:no    3:only if iteration ends   (input 1,2 or 3) :\n");
    int show_values;
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

    // <<<<<<<<<<<<<<データの読み込みと学習>>>>>>>>>>>>>>>

    double test_scores[iter+1]; // テストデータのスコアを格納する、交差検証用の配列
    int test_index[b_size*test_iter];    // テストデータの元データにおけるインデックスを格納する配列

    // 交差検証開始(CRVL は交差検証の何フェーズ目かを表す)
    for (int CRVL = 0; CRVL < (iter)+1; CRVL++) {  
    
    // ニューラルネットワークを初期化する(関数化済み)
    network_1layer *neural_network = init_network(o_dim, v_dim, b_size, D, N, fp_log, program_confirmation);

    // ニューラルネットワークをネットワークグラフで表示する
    if (show_values == 1) {
        print_network(neural_network, fp_log);
    }

    // テストデータの説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
    double **x_test = (double **)malloc(sizeof(double *) * (b_size) * (test_iter));  // test_sizeにするとエラーが出る(当たり前)
    double **y_test = (double **)malloc(sizeof(double *) * (b_size) * (test_iter));
    for (int b = 0; b < (b_size) * (test_iter); b++) {
        x_test[b] = (double *)malloc(sizeof(double) * o_dim);
        y_test[b] = (double *)malloc(sizeof(double) * v_dim);
    }
    
    if (program_confirmation == 1) {
        fprintf(fp_log, "The test data has been initialized.\n");
    }

    // データを読み込みながら、モデルの学習を行う
    // x_test, y_testにはテストデータが格納され、neural_networkの重みが更新される
    load_learn(total_epoch, iter, CRVL, b_size, test_iter, x_test, y_test, o_dim, o_col, v_dim, v_col, neural_network, alpha, leakly_relu, leakly_relu_grad, program_confirmation, show_values, fprint_graph, fp, fp_log, fp_graph);

    // テストデータを当てはめた時の決定係数を計算し表示する関数
    double r2 = test(x_test, y_test, b_size, test_iter, v_dim, neural_network, leakly_relu, leakly_relu_grad, program_confirmation, show_values, fp_log, fp_graph);

    // テストデータの決定係数を配列に格納する
    test_scores[CRVL] = r2;

    // メモリの解放
    rewind_network(neural_network); // ポインタを末尾から先頭に戻す
    free_network(neural_network);

    } // 交差検証終了

    if (fprint_graph == 1) {
        fprintf(fp_graph, "quit\n");
        pclose(fp_graph);
    }

    // 交差検証の結果を表示する
    fprintf(fp_log, "--------Review of the coefficient of determination of test data:------------\n");
    double test_score_sum = 0.0;
    for (int CRVL = 0; CRVL < (iter)+1; CRVL++) {
        fprintf(fp_log, "The coefficient of determination of test data in Cross VaLidation #%d is %f.\n", CRVL+1, test_scores[CRVL]);
        test_score_sum += test_scores[CRVL];
    }
    fprintf(fp_log, "The average of the coefficient of determination of test data is %f.\n", test_score_sum / ((iter)+1));
    printf("\nThe average of the coefficient of determination of test data is %f.\n", test_score_sum / ((iter)+1));

    return 0;
}
