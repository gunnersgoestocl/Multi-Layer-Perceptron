#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "init.h"
// #include "prop.h"

double calc_r2(double **y, double **o, int b_size, int v_dim);

// main関数
int main(int argc, char **argv) {

    // 学習率
    double alpha = 0.00001;

    // コマンドラインから分析したいファイル名を読み込む
    char *filename = argv[1];
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Cannot open %s\n", filename);
        exit(1);
    }

    // ファイルの1行目を';'で区切って、変数名を読み込み表示する
    char buf[256];
    fgets(buf, sizeof(buf), fp);
    char *token = strtok(buf, ";");
    while (token != NULL) {
        printf("%s\n", token);
        token = strtok(NULL, ";\n");
    }

    // 目的変数名の取得
    // 目的変数の変数名をコマンドラインに書き込むよう指示する
    printf("Input the name of the objective variable: ");
    char objective[256];
    scanf("%[^\n]%*1[\n]", objective);
    printf("You input %s.\n", objective);
    // 目的変数の変数名が含まれる列番号を調べる
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);
    token = strtok(buf, ";\n");
    const int v_dim = 1;  // 目的変数の個数(変数にする実装には今回はしない)
    int v_col = -1;   // 目的変数の列番号
    int col = 0;    
    while (token != NULL) {
        // printf("comparing with %s: %d\n", token, strcmp(token, objective));
        if (strcmp(token, objective) == 0) {
            v_col = col;
            printf("register success\n");
            break;
        }
        token = strtok(NULL, ";\n");
        col++;
    }
    // 目的変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
    while (v_col == -1) {
        printf("Cannot find the objective variable. Please input the correct name of the objective variable.\n");
        char objective[256];
        scanf("%[^\n]%*1[\n]", objective);
        printf("You input %s.\n", objective);
        // 目的変数の変数名が含まれる列番号を調べる
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;    
        while (token != NULL) {
            if (strcmp(token, objective) == 0) {
                v_col = col;
                printf("register success\n");
                break;
            }
            token = strtok(NULL, ";\n");
            col++;
        }
    }

    // 説明変数名の取得
    // 説明変数の変数名を一つずつコマンドラインに書き込むよう指示する
    // ALLと書いてEnterを押すと、目的変数以外全ての列を説明変数として処理する
    // 何も書かずにEnterを押すと、説明変数の入力を終了する     
    printf("Input the name of the explanatory variables (ALL to select all or END to quit selecting): ");
    char explanatory[256];
    scanf("%[^\n]%*1[\n]", explanatory);
    printf("You input %s.\n", explanatory);
    int o_dim = 0;  // 説明変数の個数
    int *o_col = (int *)malloc(sizeof(int));    // 説明変数の列番号を格納する配列
    col = 0;    // 列番号のカーソル
    while (strcmp(explanatory, "END") != 0 || o_dim == 0) {
        // 説明変数の列番号を調べる
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;

        while (token != NULL) {
            // printf("comparing with %s: %d\n", token, strcmp(token, explanatory));

            // ALLと書いた場合(目的変数以外全ての列を説明変数として処理する)
            if (strcmp(explanatory, "ALL") == 0) {
                if (strcmp(token, objective) != 0) {
                    o_col = (int *)realloc(o_col, sizeof(int) * (o_dim + 1));   // 説明変数の列番号を格納する配列のサイズを1増やす
                    o_col[o_dim] = col;
                    // col++;
                    o_dim++;
                }
                // printf("You input ALL.\n");
                // token = strtok(NULL, ";\n");
            }

            // ENDを入力した場合(説明変数の個数が0の場合を除いて、説明変数の入力を終了する)
            else if (strcmp(explanatory, "END") == 0) {
                // 説明変数の個数が0の場合、説明変数を少なくとも一つ入力するよう指示する
                if (o_dim == 0) {
                    printf("Please input at least one explanatory variable.\n");
                    break;
                }
                break;
            }

            // 何かしらの変数名を書いた場合
            else if (strcmp(token, explanatory) == 0) {
                // 列番号に重複がある場合は重複していると表示する
                printf("v_col: %d, token: %s\n", v_col, token);
                int flag = 0;
                // 目的変数と説明変数の列番号が重複している場合は、重複していると表示する
                if (col == v_col) {
                    printf("The column number %d is already registered as objective variables.\n", col);
                    flag = 1;
                    col = 0;
                    break;
                }
                for (int i = 0; i < o_dim; i++) {
                    if (o_col[i] == col) {
                        printf("The column number %d is already registered as explanatory variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    }
                }
                // 説明変数の列番号を格納する配列に、説明変数の列番号を追加する
                if (flag == 0) {
                    // 配列サイズを1増やす
                    if (o_dim != 0) {
                        o_col = (int *)realloc(o_col, sizeof(int) * (o_dim + 1));
                    }
                    o_col[o_dim] = col;
                    o_dim++;
                    printf("%s is successfully registered.\n", explanatory);
                    col = 0;    // カーソルを先頭に戻す
                    printf("o_dim: %d, o_col:", o_dim);
                    for (int i = 0; i < o_dim; i++) {
                        printf("%d ", o_col[i]);
                    }
                    printf("\n");
                }
                break;
            }
            token = strtok(NULL, ";\n");
            col++;
        }   // while (token != NULL) の終わり(変数名の探索の終わり)

        // 説明変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
        if (o_dim == 0) {
            printf("Cannot find the explanatory variable. Please input the correct name of the explanatory variable.\n");
            // break;
        }
    
        // 説明変数の列番号を昇順にソートする
        for (int i = 0; i < o_dim - 1; i++) {
            for (int j = i + 1; j < o_dim; j++) {
                if (o_col[i] > o_col[j]) {
                    int tmp = o_col[i];
                    o_col[i] = o_col[j];
                    o_col[j] = tmp;
                }
            }
        }

        // ALL or 説明変数が1つ以上ある中でのENTER の場合、説明変数の入力を終える
        if ((strcmp(explanatory, "END") == 0 && o_dim != 0)|| strcmp(explanatory, "ALL") == 0) {
            break;
        }    

        // それ以外の場合は、次の説明変数を入力する
        printf("Input the name of the explanatory variables (ALL to select all or END to quit selecting): ");
        scanf("%[^\n]%*1[\n]", explanatory);   
        printf("You input %s.\n", explanatory);
    }   // while (strcmp(explanatory, "") != 0) の終わり(説明変数の入力の終わり)

    // 目的変数と説明変数の列番号を表示し、ニューラルネットワークの設計を開始すると宣言する
    printf("The column number of the objective variable is %d.\n", v_col);
    printf("The column number of the explanatory variables are ");
    for (int i = 0; i < o_dim; i++) {
        printf("%d ", o_col[i]);
    }
    printf(".\n");
    printf("Start designing the neural network.\n");

    // ニューラルネットワークの設計
    // ニューラルネットワークの層の数、各層のニューロンの個数をスペース区切りでコマンドラインから読み込む
    printf("Input the number of layers and the number of neurons in each layer: ");
    int D;  // ニューラルネットワークの層の数
    scanf("%d", &D);
    int *N = (int *)malloc(sizeof(int) * D);    // 各層のニューロンの個数を格納する配列
    for (int i = 0; i < D; i++) {
        scanf("%d", &N[i]);
    }

    // ニューラルネットワークの層の数と各層のニューロンの個数を表示する
    printf("The number of  neuron layers is %d.\n", D+2);
    printf("The number of neurons in each layer are \n");
    printf("    input layer: %7d\n", o_dim);
    for (int i = 0; i < D; i++) {
        printf("    %2d middle layer: %3d\n", i+1, N[i]);
    }
    printf("    output layer: %7d\n", v_dim);   // 今回は出力層のニューロンは1つとあらかじめて決めてしまう
    printf("Now Loading Data File...\n");

    // データサイズの取得
    // データの行数を数える
    int data_size = -1; // 1行目は変数名なので、データの行数は1つ少ない
    while (fgets(buf, sizeof(buf), fp) != NULL) {
        data_size++;
    }
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);    // この時点でfpは2行目を指している(データの1行目を読み飛ばした)
    printf("The number of data is %d.\n", data_size);

    // バッチサイズとイテレーション数をコマンドラインから読み込む
    printf("Input the batch size and the number of iterations: ");
    int b_size; // バッチサイズ
    int iter;   // イテレーション数
    scanf("%d %d", &b_size, &iter);
    printf("The batch size is %d.\n", b_size);
    printf("The number of iterations is %d.\n", iter);
    int test_size = data_size - b_size * iter;   // テストデータのサイズ
    int test_iter = (int) (test_size / b_size);  // テストデータのイテレーション数
    // int test_iter = 1;
    printf("So, the size of test data is %d.\n", b_size * test_iter);

    double test_scores[iter+1]; // テストデータのスコアを格納する配列

    // 交差検証開始
    for (int epoch = 0; epoch < iter+1; epoch++) {  
    
    // ニューラルネットワークを初期化する
    // ニューラルネットワークの層を表す構造体の線形リストの先頭へのポインタを作る
    network_1layer *neural_network = NULL;
    // ニューラルネットワークの入力層を表すニューロン層を作る
    neuron_layer *o_layer = init_neuron_layer(0, o_dim, b_size);
    // ニューラルネットワークの中間層を表すニューロン層およびネットワーク層の線形リストを作る
    for (int i=0; i < D; i++){
        neuron_layer *v_layer = init_neuron_layer(i+1, N[i], b_size);
        neural_network = push_network_back(o_layer, v_layer, neural_network);
        o_layer = v_layer;
    }
    // ニューラルネットワークの出力層を表すニューロン層を作る
    neuron_layer *v_layer = init_neuron_layer(D+1, v_dim, b_size);
    neural_network = push_network_back(o_layer, v_layer, neural_network);    // 第 D+1 層まで初期化されたニューラルネットワークを表す構造体の線形リストができた
    printf("The neural network has been initialized.\n");
    // ここまでエラーなし 12/30 22:25

    // テストデータの読み込み
    // テストデータの説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
    double **x_test = (double **)malloc(sizeof(double *) * b_size * test_iter);  // test_sizeにするとエラーが出る(当たり前)
    double **y_test = (double **)malloc(sizeof(double *) * b_size * test_iter);
    for (int b = 0; b < b_size * test_iter; b++) {
        x_test[b] = (double *)malloc(sizeof(double) * o_dim);
        y_test[b] = (double *)malloc(sizeof(double) * v_dim);
    }
    int test_flag = 0;  // テストデータの読み込みが完了したら1にする
    printf("The test data has been initialized.\n");

    // データを読み込んでニューラルネットワークを学習させる
    for (int iter_num = 0; iter_num<iter; iter_num++) {

        // テストデータの読み込み
        if (iter_num == epoch && test_flag == 0) {
            // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
            for (int b = 0; b < b_size*test_iter; b++) {
                fgets(buf, sizeof(buf), fp);
                token = strtok(buf, ";\n");
                col = 0;
                while (token != NULL) {
                    for (int i = 0; i < o_dim; i++) {
                        if (col == o_col[i]) {
                            x_test[b][i] = atof(token);
                        }
                    }
                    if (col == v_col) {
                        y_test[b][0] = atof(token);
                    }
                    token = strtok(NULL, ";\n");
                    col++;
                }
            }   // 1バッチのデータ読み込み完了
            printf("The test data has been loaded.\n");
            test_flag = 1;
            iter_num--;
        }

        // 訓練データの読み込み
        else {
        // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
        double **x = (double **)malloc(sizeof(double *) * b_size);
        double **y = (double **)malloc(sizeof(double *) * b_size);
        for (int b = 0; b < b_size; b++) {
            x[b] = (double *)malloc(sizeof(double) * o_dim);
            y[b] = (double *)malloc(sizeof(double) * v_dim);
        }
        // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に格納する
        for (int b = 0; b < b_size; b++) {
            fgets(buf, sizeof(buf), fp);
            token = strtok(buf, ";\n");
            col = 0;
            while (token != NULL) {
                for (int i = 0; i < o_dim; i++) {
                    if (col == o_col[i]) {
                        x[b][i] = atof(token);
                    }
                }
                if (col == v_col) {
                    y[b][0] = atof(token);
                }
                token = strtok(NULL, ";\n");
                col++;
            }
        }   // 1バッチのデータ読み込み完了
        printf("The data of iteration %d has been loaded.\n", iter_num+1);
        // // xの値を表示する
        // for (int b = 0; b < b_size; b++) {
        //     for (int i = 0; i < o_dim; i++) {
        //         printf("%f ", x[b][i]);
        //     }
        //     printf("\n");
        // }
        // ここまでエラーなし 12/30 22:27

        // 順伝播を行う
        neural_network = forward_prop(x, y, neural_network, leakly_relu, leakly_relu_grad);
        printf("The forward propagation of iteration %d has been completed.\n", iter_num+1);
        // printf("neural_network is at %dth layer.\n", neural_network -> o_layer -> k);

        // // 出力の推定量を二次元配列に格納する
        // double **y_hat = (double **)malloc(sizeof(double *) * b_size);
        // for (int b = 0; b < b_size; b++) {
        //     y_hat[b] = (double *)malloc(sizeof(double) * v_dim);
        //     for (int i = 0; i < v_dim; i++) {
        //         y_hat[b][i] = neural_network -> v_layer -> neuron_2darray[b][i].o;
        //     }
        // }
        // この内容は関数にする
        double **y_hat = estimate(b_size, v_dim, neural_network);

        printf("The estimation of iteration %d has been completed. Deviation is shown below.\n", iter_num+1);
        // y_hatの値を表示する
        for (int b = 0; b < b_size; b++) {
            for (int i = 0; i < v_dim; i++) {
                printf("%f ", y_hat[b][i] - y[b][i]);
            }
            // printf("\n");
        }
        printf("\n");
        // ここまでエラーなし 12/30 22:53

        // 誤差逆伝播を行いニューラルネットワークを更新する
        neural_network = back_prop(neural_network, alpha, b_size);
        printf("The back propagation of iteration %d has been completed.\n", iter_num+1);

        // 1イテレーションごとの決定係数を計算し表示する
        double r2 = calc_r2(y, y_hat, b_size, v_dim);
        printf("The coefficient of determination of iteration %d is %f.\n", iter_num+1, r2);
        // neural_network = rewind_network(neural_network);   // ポインタを末尾から先頭に戻す
        }   // else(訓練データを読んだ場合)の終わり
    }

    // // テストデータの読み込み
    // // テストデータの説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
    // double **x_test = (double **)malloc(sizeof(double *) * b_size);  // test_sizeにするとエラーが出る(当たり前)
    // double **y_test = (double **)malloc(sizeof(double *) * b_size);
    // for (int b = 0; b < b_size; b++) {
    //     x_test[b] = (double *)malloc(sizeof(double) * o_dim);
    //     y_test[b] = (double *)malloc(sizeof(double) * v_dim);
    // }
    if (test_flag == 0) {
        // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
        for (int b = 0; b < b_size*test_iter; b++) {
            fgets(buf, sizeof(buf), fp);
            token = strtok(buf, ";\n");
            col = 0;
            while (token != NULL) {
                for (int i = 0; i < o_dim; i++) {
                    if (col == o_col[i]) {
                        x_test[b][i] = atof(token);
                    }
                }
                if (col == v_col) {
                    y_test[b][0] = atof(token);
                }
                token = strtok(NULL, ";\n");
                col++;
            }
        }   // 1バッチのデータ読み込み完了
        printf("The test data has been loaded.\n");
    }
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);    // この時点でfpは2行目を指している(データの1行目を読み飛ばした)


    // テストデータを当てはめた時の決定係数を計算し表示する
    // printf("The test data has been loaded.\n");
    printf("Now calculating the coefficient of determination of test data...\n");
    neural_network = rewind_network(neural_network);   // ポインタを末尾から先頭に戻す
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    double **y_hat = (double **)malloc(sizeof(double *) * b_size * test_iter);
    for (int b = 0; b < b_size * test_iter; b++) {
        y_hat[b] = (double *)malloc(sizeof(double) * v_dim);
    }
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    for (int iter_num = 0; iter_num < test_iter; iter_num++) {
        neural_network = forward_prop(&x_test[iter_num*b_size], &y_test[iter_num*b_size], neural_network, leakly_relu, leakly_relu_grad);
        // for (int b = 0; b < b_size; b++) {
        //     for (int i = 0; i < v_dim; i++) {
        //         y_hat[iter_num*b_size+b][i] = neural_network -> v_layer -> neuron_2darray[b][i].o;
        //     }
        // }
        // 関数に置き換えると
        estimate_sub(b_size, iter_num, v_dim, neural_network, y_hat);

    }


    for (int b = 0; b < b_size*test_iter; b++) {
        for (int i = 0; i < v_dim; i++) {
            printf("%f ", y_hat[b][i]);
        }
        // printf("\n");
    }
    printf("\n");

    double r2 = calc_r2(y_test, y_hat, b_size*test_iter, v_dim);
    printf("The coefficient of determination of test data is %f.\n", r2);
    // テストデータの決定係数を配列に格納する
    test_scores[epoch] = r2;

    // メモリの解放
    rewind_network(neural_network); // ポインタを末尾から先頭に戻す
    free_network(neural_network);

    } // 交差検証終了

    // 交差検証の結果を表示する
    printf("--------Review of the coefficient of determination of test data:------------\n");
    double test_score_sum = 0.0;
    for (int epoch = 0; epoch < iter+1; epoch++) {
        printf("The coefficient of determination of test data in epoch %d is %f.\n", epoch+1, test_scores[epoch]);
        test_score_sum += test_scores[epoch];
    }
    printf("The average of the coefficient of determination of test data is %f.\n", test_score_sum / (iter+1));

    return 0;
}


// 8. 決定係数を計算する関数```calc_r2```を定義する
//    - 引数には、目的変数値の二次元配列```y```と、出力の推定量の二次元配列```o```をとる
//    - 各行ごとに決定係数を計算し、その平均値を返す
//    - 決定係数は、線形回帰ではないので、 で計算する

double calc_r2(double **y, double **o, int b_size, int v_dim) {
    double r2 = 0.0;

    // 今回は出力が1次元の場合のみを想定するため、ループも一重であるし、```v_dim```も使わない
    double y_sum = 0.0;
    // double o_sum = 0.0;
    double y_mean = 0.0;
    // double o_mean = 0.0;
    double y_var = 0.0;
    // double o_var = 0.0;
    // double cov = 0.0;
    double sq_hensa = 0.0;
    for (int b = 0; b < b_size; b++) {
        y_sum += y[b][0];   // 出力が1次元でない場合は要変更
        // o_sum += o[b][0];
    }
    // o_mean = o_sum / b_size;
    y_mean = y_sum / b_size;
    for (int b = 0; b < b_size; b++) {
        y_var += pow(y[b][0] - y_mean, 2);
        // o_var += pow(o[b][0] - o_mean, 2);
        sq_hensa += pow(y[b][0] - o[b][0], 2);
    }
    r2 = 1 - sq_hensa / y_var;
    
    return r2;
}

