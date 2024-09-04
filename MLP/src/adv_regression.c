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

    /*
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
    */

    config *setting = set_config();     // 分析対象・記録内容・タスクの設定  Setting of analysis target, recording content, and task
    set_variables(setting);             // 変数の設定  Variable setting
    model *model = set_model(setting);  // モデルの設定  Model setting
    print_model(setting, model);
    set_learning(setting);              // 学習の設定  Learning setting
    run(setting, model);                // 学習以降のプログラムを実行  Execute the program after learning
    printf("Program is completed.\n");

    return 0;
}
