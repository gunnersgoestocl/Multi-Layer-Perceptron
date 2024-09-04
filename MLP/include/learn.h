/*learn.h*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "init.h"
#include "setting.h"
#include "eval_func.h"

#ifndef _LEARN_H_
#define _LEARN_H_

/* ニューラルネットワークの初期化 */
network_1layer *init_network(config *setting, model *M);

/* テストデータを読み込む (テストデータの位置に対処するためのフラグ操作が存在) */
void load_testData(config *setting, int *test_flag, double **x_test, double **y_test);

/* バッチサイズ分のデータを読み込む */
void load_Data(config *setting, double **x, double **y);

/* 学習済みのモデルについて、テストデータを与えて、評価値を返す */
double test(config *setting, double **x_test, double **y_test, network_1layer *neural_network);

/* 設計したモデルについて、初期化・データの読み込み・学習・テストを行い、評価値を返す */
double learn_set(config *setting, model *M);

/* 設計したモデルについて、データを与えて、学習を行い、学習中の評価値の推移を出力し、必要ならテストデータを抽出する */
void load_learn(config *setting, model *M, network_1layer *neural_network, double **x_test, double **y_test);

// 設計したモデルについて、データを与えて、交差検証法により性能を評価する
double CRVL(config *setting, model *M);

/* 学習以降のプログラムを実行する関数 */
void run(config *setting, model *M);

#endif