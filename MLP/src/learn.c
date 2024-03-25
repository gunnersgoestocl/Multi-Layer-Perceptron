#include "learn.h"

network_1layer  *init_network(int o_dim, int v_dim, int b_size, int D, int *N, FILE *fp_log, int program_confirmation){
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
    if (program_confirmation == 1) {
        fprintf(fp_log, "The neural network has been initialized.\n");
    }
    return neural_network;
    // fprintf(fp_log, "The neural network has been initialized.\n");
    // ここまでエラーなし 12/30 22:25
}

void load_testData(FILE *fp, int *test_flag, int b_size, int test_iter, double **x_test, double **y_test, int o_dim, int *o_col, int v_dim, int *v_col){
    // int col = 0;    // 列番号のカーソル // 変数名捜索のためのcolは関数内に格納するため、外出しにする
    // char buf[256];  // ファイルの1行を読み込むためのバッファ
    // char *token;   // 上の方の定義は関数内に格納するため、ここで定義する
    if (*test_flag == 0) {
        // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
        load_Data(fp, b_size*test_iter, x_test, y_test, o_dim, o_col, v_dim, v_col);    // 1バッチのデータ読み込み完了
        
        // if (program_confirmation == 1) {
        //     fprintf(fp_log, "The test data has been loaded.\n");
        // }
        *test_flag = 1;
    }
}

void load_Data(FILE *fp, int b_size, double **x, double **y, int o_dim, int *o_col, int v_dim, int *v_col){
    int col = 0;    // 列番号のカーソル // 変数名捜索のためのcolは関数内に格納するため、外出しにする
    char buf[256];  // ファイルの1行を読み込むためのバッファ
    char *token;   // 上の方の定義は関数内に格納するため、ここで定義する
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
            for (int i = 0; i < v_dim; i++) {
                if (col == v_col[i]) {
                    y[b][i] = atof(token);
                }
            }
            // if (col == v_col) {
            //     y[b][0] = atof(token);
            // }
            token = strtok(NULL, ";\n");
            col++;
        }
    }   // 1バッチのデータ読み込み完了
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

// 設計したモデルについて、データを与えて、交差検証法により、学習とテストを両方行う関数
void CRVL(int total_epoch, int iter, int b_size, int test_iter, int D, int *N, int o_dim, int *o_col, int v_dim, int *v_col, double alpha, double (*activator)(double), double (*activator_grad)(double), int program_confirmation, int show_values, int fprint_graph, FILE *fp, FILE *fp_log, FILE *fp_graph) {
    double test_scores[iter+1]; // テストデータのスコアを格納する、交差検証用の配列

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

    // 交差検証の結果を表示する
    fprintf(fp_log, "--------Review of the coefficient of determination of test data:------------\n");
    double test_score_sum = 0.0;
    for (int CRVL = 0; CRVL < (iter)+1; CRVL++) {
        fprintf(fp_log, "The coefficient of determination of test data in Cross VaLidation #%d is %f.\n", CRVL+1, test_scores[CRVL]);
        test_score_sum += test_scores[CRVL];
    }
    fprintf(fp_log, "The average of the coefficient of determination of test data is %f.\n", test_score_sum / ((iter)+1));
    printf("\nThe average of the coefficient of determination of test data is %f.\n", test_score_sum / ((iter)+1));
}

// データを読み込みながら、モデルの学習を行う関数
// 引数は、(エポック数、ミニバッチ数、テストデータの読み込みタイミング、バッチサイズ、テストデータのバッチ数、テストデータの説明変数配列、テストデータの目的変数配列、説明変数の次元、説明変数の列番号配列、目的変数の次元、目的変数の列番号配列、ニューラルネットワークのポインタ、学習率、活性化関数、活性化関数の微分、プログラム確認フラグ、値表示フラグ、グラフ表示フラグ、データファイルポインタ、ログファイルポインタ、グラフファイルポインタ)
void load_learn(int total_epoch, int iter, int CRVL, int b_size, int test_iter, double **x_test, double **y_test, int o_dim, int *o_col, int v_dim, int *v_col, network_1layer *neural_network, double alpha, double (*activator)(double), double (*activator_grad)(double), int program_confirmation, int show_values, int fprint_graph, FILE *fp, FILE *fp_log, FILE *fp_graph) {
    char buf[256];  // ファイルの1行目を読み飛ばすためのバッファ
    // テストデータの読み込み
    int test_flag = 0;  // テストデータの読み込みが完了したら1にする

    // 訓練データに対する決定係数の推移を追うための配列を初期化する
    double *r2_epoch = (double *)malloc(sizeof(double) * (total_epoch));

    // データを読み込んでニューラルネットワークを学習させる

    // エポックループ
    for (int epoch=0; epoch < total_epoch; epoch++) {
        fprintf(fp_log, "\n===================================================\n");
        fprintf(fp_log, "The epoch %d has started.\n", epoch+1);


    // ミニバッチを取り出し、順伝播、誤差逆伝播を行う
    for (int iter_num = 0; iter_num < iter; iter_num++) {

        // テストデータの読み込み(データセットの最後の部分をテストデータにしない場合)
        if (iter_num == CRVL && test_flag == 0) {
            // テストデータの読み込みを関数化(テストデータの説明変数の値を```x_test```、目的変数の値を```y_test```に格納する)
            load_testData(fp, &test_flag, b_size, test_iter, x_test, y_test, o_dim, o_col, v_dim, v_col);   // 1バッチのデータ読み込み完了
            iter_num--;     // これはtest_flag == 0の時しか発動させてはいけない

            if (program_confirmation == 1) {
                fprintf(fp_log, "The test data has been loaded.\n");
            }
        }

        // 訓練データの読み込み
        else {
        // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
        double **x = (double **)malloc(sizeof(double *) * (b_size));
        double **y = (double **)malloc(sizeof(double *) * (b_size));
        for (int b = 0; b < b_size; b++) {
            x[b] = (double *)malloc(sizeof(double) * o_dim);
            y[b] = (double *)malloc(sizeof(double) * v_dim);
        }
        // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に格納する
        load_Data(fp, b_size, x, y, o_dim, o_col, v_dim, v_col);     // 1バッチのデータ読み込み完了

        if (program_confirmation == 1) {
            fprintf(fp_log, "The data of epoch %d iteration %d has been loaded.\n", epoch+1, iter_num+1);
        }

        // 順伝播を行う
        neural_network = forward_prop(fp_log, x, y, neural_network, leakly_relu, leakly_relu_grad, program_confirmation, show_values);
        if (program_confirmation == 1) {
            fprintf(fp_log, "The forward propagation of epoch %d iteration %d has been completed.\n", epoch+1, iter_num+1);
        }
        // fprintf(fp_log, "The forward propagation of epoch %d iteration %d has been completed.\n", epoch+1, iter_num+1);
        //printf("neural_network is at %dth layer.\n", neural_network -> o_layer -> k);

        // この内容は関数にする
        double **y_hat = estimate(b_size, v_dim, neural_network);

        if (program_confirmation == 1) {
            fprintf(fp_log, "The estimation of epoch %d iteration %d has been completed. Deviation is shown below.\n", epoch+1, iter_num+1);
            // y_hat(モデルによる推定値)とy(正解データ)との値の差を表示する
            for (int b = 0; b < b_size; b++) {
                for (int i = 0; i < v_dim; i++) {
                    fprintf(fp_log, "%f ", y_hat[b][i] - y[b][i]);
                }
                // printf("\n");
            }
            fprintf(fp_log, "\n");
        }
        // ここまでエラーなし 12/30 22:53

        // 誤差逆伝播を行いニューラルネットワークを更新する
        neural_network = back_prop(fp_log, neural_network, alpha, b_size, program_confirmation, show_values);
        if (program_confirmation == 1) {
            fprintf(fp_log, "The back propagation of epoch %d iteration %d has been completed.\n", epoch+1, iter_num+1);
        }

        // 1イテレーションごとの決定係数を計算し表示する
        double r2 = calc_r2(y, y_hat, b_size, v_dim);
        fprintf(fp_log, "The coefficient of determination of epoch %d iteration %d is %f.\n", epoch+1, iter_num+1, r2);

        // エポック終了時には、そのエポックの決定係数を配列に格納する
        if (iter_num == (iter)-1) {
            r2_epoch[epoch] = r2;
        }

        }   // else(訓練データを読んだ場合)の終わり
        
    }   // ミニバッチを取り出し、順伝播、誤差逆伝播を行うループの終わり


    // テストデータの読み込み(データセットの最後の部分をテストデータにする場合)
    if (test_flag == 0) {
        // // テストデータの説明変数の値を```x_test```、目的変数の値を```y_test```に格納する
        load_testData(fp, &test_flag, b_size, test_iter, x_test, y_test, o_dim, o_col, v_dim, v_col);   // 1バッチのデータ読み込み完了
        if (program_confirmation == 1) {
            fprintf(fp_log, "The test data has been loaded.\n");
        }
        // fprintf(fp_log, "The test data has been loaded.\n");
    }
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);    // この時点でfpは2行目を指している(データの1行目を読み飛ばした)

    }   // エポックループの終わり

    // 学習時の決定係数の推移をグラフで表示する
    if (fprint_graph == 1) {
        for (int i = 0; i < (total_epoch); i++) {
            fprintf(fp_graph, "%d %f\n", i+1, r2_epoch[i]);
        }
        fprintf(fp_graph, "e\n");
    }
    ///////////////////////////////////この間を関数化する
}


// テストデータを当てはめた時の決定係数を計算し表示する関数
double test(double **x_test, double **y_test, int b_size, int test_iter, int v_dim, network_1layer *neural_network, double (*activator)(double), double (*activator_grad)(double), int program_confirmation, int show_values, FILE *fp_log, FILE *fp_graph){
    if (program_confirmation == 1) {
        fprintf(fp_log, "Now calculating the coefficient of determination of test data...\n");
    }
    
    neural_network = rewind_network(neural_network);   // ポインタを末尾から先頭に戻す
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    double **y_hat = (double **)malloc(sizeof(double *) * (b_size) * (test_iter));
    for (int b = 0; b < (b_size) * (test_iter); b++) {
        y_hat[b] = (double *)malloc(sizeof(double) * v_dim);
    }
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    for (int iter_num = 0; iter_num < test_iter; iter_num++) {
        neural_network = forward_prop(fp_log, &x_test[iter_num*(b_size)], &y_test[iter_num*(b_size)], neural_network, leakly_relu, leakly_relu_grad, program_confirmation, show_values);
        estimate_sub(b_size, iter_num, v_dim, neural_network, y_hat);
    }

    // テストデータに対する当てはめ値を表示する
    fprintf(fp_log, "The value estimated from test data is below : \n");
    for (int b = 0; b < (b_size)*(test_iter); b++) {
        for (int i = 0; i < v_dim; i++) {
            fprintf(fp_log, "%f ", y_hat[b][i]);
        }
        // printf("\n");
    }
    fprintf(fp_log, "\n");

    // テストデータの正解値と当てはめ値から決定係数を計算し表示する
    double r2 = calc_r2(y_test, y_hat, (b_size)*(test_iter), v_dim);
    fprintf(fp_log, "The coefficient of determination of test data is %f.\n", r2);

    return r2;
}