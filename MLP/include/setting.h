#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "activator.h"
#include "loss_func.h"
#include "eval_func.h"

#ifndef _SETTING_H_
#define _SETTING_H_

/* 分析対象・記録内容・タスクの設定を格納する構造体 */
typedef struct config_ config;
config *set_config();
void set_variables(config *setting);
char *get_filename(const char *path);   // pathからファイル名を取得する関数

/* モデルの設定を格納する構造体 */
typedef struct model_ model;
model *set_model(config *setting);
void print_model(config *setting, model *model);

/* 学習の設定 */
void set_learning(config *setting);

/* 構造体の定義 */
struct config_ {
    FILE *fp;       // データファイルのポインタ
    FILE *fp_log;   // ログファイルのポインタ
    FILE *fp_graph; // グラフファイルのポインタ

    int v_dim;      // 目的変数の個数(出力層の次元)
    int *v_col;     // 目的変数の列番号を格納する配列
    int o_dim;      // 説明変数の個数(入力層の次元)
    int *o_col;     // 説明変数の列番号を格納する配列

    int task_type;      // タスクの種類 (0: 回帰, 1: 分類)
    int *v_class;       // 出力層の各次元のクラス数 (分類問題の場合)

    int data_size;      // データの行数
    int b_size;         // バッチサイズ
    int iter;           // イテレーション数
    int test_size;      // テストデータのサイズ
    int test_iter;      // テストデータのイテレーション数
    int total_epoch;    // エポック数
    double alpha;       // 学習率
    int loss_func;      // 損失関数の種類 (0: MSE, 1: BCE, 2: CCE, 3: KLdivergence)
    model *MODEL;       // ニューラルネットワークのモデル

    int test_train;     // モデルの性能評価とモデルの学習のどちらが目的か (0: 学習, 1: 性能評価)
    double (*eval_func)(double **, double **, int, int); // 性能評価関数の関数ポインタ (R2, accuracy)

    int cross_val;      // クロスバリデーションを行うかのフラグ (0: 行わない;ホールドアウト, 1: 行う)
    int test_pos;       // ホールドアウト法における、テストデータの位置
    int program_log;    // プログラムの処理内容を詳細に出力するかのフラグ
    int show_values;    // 各ステージでのニューロンの値を記録するかのフラグ
    int fprint_graph;   // グラフをファイルに出力するかのフラグ
};

struct model_ {
    int o_dim;      // 入力層の次元
    int v_dim;      // 出力層の次元 (labelの数)
    int D;          // ニューラルネットワークの隠れ層の数
    int *N;         // 各隠れ層のニューロンの数
    int *v_class;       // 出力層の各次元のクラス数 (分類問題の場合)
    int v_class_sum;    // 出力層のクラス数の合計

    /* 0:linear, 1:ReLU, 2:Leakly_ReLU, 3:Sigmoid, 4:tanh */
    int *activator;     // 各層の活性化関数の種類
    int out_activator;  // 出力層の活性化関数の種類

    int b_size;     // バッチサイズ (再学習の際に使用)
};


#endif