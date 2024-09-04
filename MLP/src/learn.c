#include "learn.h"

network_1layer *init_network(config *setting, model *M){
    // ニューラルネットワークの層を表す構造体の線形リストの先頭へのポインタを作る
    network_1layer *neural_network = NULL;
    // ニューラルネットワークの入力層を表すニューロン層を作る
    neuron_layer *o_layer = init_neuron_layer(0, M -> o_dim, 0, setting -> b_size);     // 入力層のactivatorは恒等関数
    // ニューラルネットワークの中間層を表すニューロン層およびネットワーク層の線形リストを作る
    for (int i=0; i < M -> D; i++){
        neuron_layer *v_layer = init_neuron_layer(i+1, M -> N[i], M -> activator[i], setting -> b_size);
        neural_network = push_network_back(o_layer, v_layer, neural_network);
        o_layer = v_layer;
    }
    // ニューラルネットワークの出力層を表すニューロン層を作る
    int N_out = M -> v_dim;     // 出力層のニューロンの数 (回帰モデル、分類(Multi-label, Multi-class含む)モデル兼用)
    if (setting -> task_type == 1) {
        N_out = M -> v_class_sum;
    }
    neuron_layer *v_layer = init_neuron_layer((M -> D)+1, N_out, M -> out_activator,setting -> b_size);
    neural_network = push_network_back(o_layer, v_layer, neural_network);    // 第 D+1 層まで初期化されたニューラルネットワークを表す構造体の線形リストができた
    if (setting -> program_log == 1) {
        fprintf(setting -> fp_log, "The neural network has been initialized.\n");
    }
    return neural_network;
}

void load_testData(config *setting, int *test_flag, double **x_test, double **y_test){
    if (*test_flag == 0) {
        // テストデータの説明変数の値を```x```、目的変数の値を```y```に格納する
        load_Data(setting, x_test, y_test);    // 1バッチのデータ読み込み完了
        
        *test_flag = 1;
    }
}

void load_Data(config *setting, double **x, double **y){
    int col = 0;    // 列番号のカーソル // 変数名捜索のためのcolは関数内に格納するため、外出しにする
    char buf[256];  // ファイルの1行を読み込むためのバッファ
    char *token;   // 上の方の定義は関数内に格納するため、ここで定義する
    // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に格納する
    for (int b = 0; b < setting -> b_size; b++) {
        fgets(buf, sizeof(buf), setting -> fp);
        token = strtok(buf, ";\n");
        col = 0;
        while (token != NULL) {
            for (int i = 0; i < setting -> o_dim; i++) {
                if (col == setting -> o_col[i]) {
                    x[b][i] = atof(token);
                }
            }
            for (int i = 0; i < setting -> v_dim; i++) {
                if (col == setting -> v_col[i]) {
                    y[b][i] = atof(token);
                }
            }
            token = strtok(NULL, ";\n");
            col++;
        }
    }   // 1バッチのデータ読み込み完了
}

// 設計したモデルについて、データを与えて、交差検証法により、学習とテストを両方行う関数
double CRVL(config *setting, model *M){
    double test_scores[(setting -> iter)+1]; // テストデータのスコアを格納する、交差検証用の配列

    // 交差検証開始(CRVL は交差検証の何フェーズ目かを表す)
    for (int CRVL = 0; CRVL < (setting -> iter)+1; CRVL++) {  
        // CRVL番目のバッチから指定された行数を、テストデータとして除いたデータを、学習データとして読み込み、モデルの学習を行い、テストデータの決定係数を計算する
        double r2 = learn_set(setting, M);

        // テストデータの決定係数を配列に格納する
        test_scores[CRVL] = r2;
    } // 交差検証終了

    // 交差検証の結果を表示する
    fprintf(setting -> fp_log, "--------Review of the coefficient of determination of test data:------------\n");
    double test_score_sum = 0.0;
    double test_score_best = -500.0;
    double test_score_worst = 1.0;
    for (int CRVL = 0; CRVL < (setting -> iter)+1; CRVL++) {
        fprintf(setting -> fp_log, "The coefficient of determination of test data in Cross VaLidation #%d is %f.\n", CRVL+1, test_scores[CRVL]);
        test_score_sum += test_scores[CRVL];
        if (test_scores[CRVL] > test_score_best) {
            test_score_best = test_scores[CRVL];
        }
        if (test_scores[CRVL] < test_score_worst) {
            test_score_worst = test_scores[CRVL];
        }
    }
    fprintf(setting -> fp_log, "The coefficient of determination of test data:(average) %f, (best) %f, (worst) %f\n", test_score_sum / ((setting -> iter)+1), test_score_best, test_score_worst);
    printf("\nThe coefficient of determination of test data:(average) %f, (best) %f, (worst) %f\n", test_score_sum / ((setting -> iter)+1), test_score_best, test_score_worst);

    return test_score_sum / ((setting -> iter)+1);
}

// 設計したモデルについて、ニューロンとネットワークの初期化、データの読み込み、学習、テストを一体として行う関数
double learn_set(config *setting, model *M){
    // ニューラルネットワークを初期化する(関数化済み)
    network_1layer *neural_network = init_network(setting, M);
    // ニューラルネットワークをネットワークグラフで表示する
    if (setting -> show_values == 1) {
        print_network(neural_network, setting -> fp_log);
    }

    // テストデータの説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
    double **x_test = (double **)malloc(sizeof(double *) * (setting -> b_size) * (setting -> test_iter));  // test_sizeにするとエラーが出る(当たり前)
    double **y_test = (double **)malloc(sizeof(double *) * (setting -> b_size) * (setting -> test_iter));
    for (int b = 0; b < (setting -> b_size) * (setting -> test_iter); b++) {
        x_test[b] = (double *)malloc(sizeof(double) * M -> o_dim);
        y_test[b] = (double *)malloc(sizeof(double) * M -> v_dim);      // 要変更 (Multi-class, Multi-label)
    }
    if (setting -> program_log == 1) {
        fprintf(setting -> fp_log, "The test data has been initialized.\n");
    }

    // データを読み込みながら、モデルの学習を行う
    // x_test, y_testにはテストデータが格納され、neural_networkの重みが更新される
    load_learn(setting, M, neural_network, x_test, y_test);

    // テストデータを当てはめた時の決定係数を計算し表示する関数
    double r2 = test(setting, x_test, y_test, neural_network);  // 要変更 (classification task)
    fprintf(setting -> fp_log, "The coefficient of determination of test data is %f.\n", r2);

    // メモリの解放
    rewind_network(neural_network); // ポインタを末尾から先頭に戻す
    free_network(neural_network);

    return r2;
}

// データを読み込みながら、モデルの学習を行う関数
void load_learn(config *setting, model *M, network_1layer *neural_network, double **x_test, double **y_test){
    char buf[256];  // ファイルの1行目を読み飛ばすためのバッファ
    // テストデータの読み込み
    int test_flag = 0;  // テストデータの読み込みが完了したら1にする

    // 訓練データに対する決定係数の推移を追うための配列を初期化する
    double *r2_epoch = (double *)malloc(sizeof(double) * (setting -> total_epoch));

    // データを読み込んでニューラルネットワークを学習させる

    // エポックループ
    for (int epoch=0; epoch < setting -> total_epoch; epoch++) {
        fprintf(setting -> fp_log, "\n===================================================\n");
        fprintf(setting -> fp_log, "The epoch %d has started.\n", epoch+1);
    

    // ミニバッチを取り出し、順伝播、誤差逆伝播を行う
    for (int iter_num = 0; iter_num < setting -> iter; iter_num++) {

        // テストデータの読み込み(データセットの最後の部分をテストデータにしない場合)
        if (iter_num == setting -> test_pos && test_flag == 0) {
            // テストデータの読み込みを関数化(テストデータの説明変数の値を```x_test```、目的変数の値を```y_test```に格納する)
            load_testData(setting, &test_flag, x_test, y_test);   // 1バッチのデータ読み込み完了
            iter_num--;     // これはtest_flag == 0の時しか発動させてはいけない

            if (setting -> program_log == 1) {
                fprintf(setting -> fp_log, "The test data has been loaded.\n");
            }
        }

        // 訓練データの読み込み
        else {
        // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に、それぞれ二次元配列のサイズを拡張した上で格納する
        double **x = (double **)malloc(sizeof(double *) * (setting -> b_size));
        double **y = (double **)malloc(sizeof(double *) * (setting -> b_size));
        for (int b = 0; b < setting -> b_size; b++) {
            x[b] = (double *)malloc(sizeof(double) * M -> o_dim);
            y[b] = (double *)malloc(sizeof(double) * M -> v_dim);
        }
        // バッチサイズ分の行を読み込むごとに、説明変数の値を```x```、目的変数の値を```y```に格納する
        load_Data(setting, x, y);     // 1バッチのデータ読み込み完了
        if (setting -> program_log == 1) {
            fprintf(setting -> fp_log, "The data of epoch %d iteration %d has been loaded.\n", epoch+1, iter_num+1);
        }

        // 順伝播を行う
        neural_network = forward_prop(setting, M, neural_network, x, y);
        if (setting -> program_log == 1) {
            fprintf(setting -> fp_log, "The forward propagation of epoch %d iteration %d has been completed.\n", epoch+1, iter_num+1);
        }

        // この内容は関数にする
        double **y_hat = estimate(setting -> b_size, M -> v_dim, neural_network);
        if (setting -> program_log == 1) {
            fprintf(setting -> fp_log, "The estimation of epoch %d iteration %d has been completed. Deviation is shown below.\n", epoch+1, iter_num+1);
            // y_hat(モデルによる推定値)とy(正解データ)との値の差を表示する
            for (int b = 0; b < setting -> b_size; b++) {
                for (int i = 0; i < M -> v_dim; i++) {
                    fprintf(setting -> fp_log, "%f ", y_hat[b][i] - y[b][i]);
                }
            }
            fprintf(setting -> fp_log, "\n");
        }

        // 誤差逆伝播を行いニューラルネットワークを更新する
        neural_network = back_prop(setting, M, neural_network, setting -> alpha, setting -> b_size);
        if (setting -> program_log == 1) {
            fprintf(setting -> fp_log, "The back propagation of epoch %d iteration %d has been completed.\n", epoch+1, iter_num+1);
        }

        // 1イテレーションごとの決定係数を計算し表示する
        double r2 = calc_r2(y, y_hat, setting -> b_size, setting -> v_dim);
        fprintf(setting -> fp_log, "The coefficient of determination of epoch %d iteration %d is %f.\n", epoch+1, iter_num+1, r2);

        // エポック終了時には、そのエポックの決定係数を配列に格納する
        if (iter_num == (setting -> iter)-1) {
            r2_epoch[epoch] = r2;
        }

        }   // else(訓練データを読んだ場合)の終わり
    }   // ミニバッチを取り出し、順伝播、誤差逆伝播を行うループの終わり

    // テストデータの読み込み(データセットの最後の部分をテストデータにする場合)
    if (test_flag == 0) {
        load_testData(setting, &test_flag, x_test, y_test);   // 1バッチのデータ読み込み完了
        if (setting -> program_log == 1) {
            fprintf(setting -> fp_log, "The test data has been loaded.\n");
        }
    }

    fseek(setting -> fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), setting -> fp);    // この時点でfpは2行目を指している(データの1行目を読み飛ばした)
    }   // エポックループの終わり

    // 学習時の決定係数の推移をグラフで表示する
    if (setting -> fprint_graph == 1) {
        for (int i = 0; i < (setting -> total_epoch); i++) {
            fprintf(setting -> fp_graph, "%d %f\n", i+1, r2_epoch[i]);
        }
        fprintf(setting -> fp_graph, "e\n");
    }
}

// テストデータを当てはめた時の決定係数を計算し表示する関数
double test(config *setting, double **x_test, double **y_test, network_1layer *neural_network){
    if (setting -> program_log == 1) {
        fprintf(setting -> fp_log, "Now calculating the coefficient of determination of test data...\n");
    }
    
    neural_network = rewind_network(neural_network);   // ポインタを末尾から先頭に戻す
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    double **y_hat = (double **)malloc(sizeof(double *) * (setting -> b_size) * (setting -> test_iter));
    for (int b = 0; b < (setting -> b_size) * (setting -> test_iter); b++) {
        y_hat[b] = (double *)malloc(sizeof(double) * (setting -> v_dim));
    }
    // テストデータをバッチサイズ分の行ずつ読み込んで、順伝播を行い、出力の推定量を二次元配列に格納する
    for (int iter_num = 0; iter_num < setting -> test_iter; iter_num++) {
        neural_network = forward_prop(setting, setting -> MODEL, neural_network, &x_test[iter_num*(setting -> b_size)], &y_test[iter_num*(setting -> b_size)]);
        estimate_sub(setting -> b_size, iter_num, setting -> v_dim, neural_network, y_hat);
    }

    // テストデータに対する当てはめ値を表示する
    fprintf(setting -> fp_log, "The value estimated from test data is below : \n");
    for (int b = 0; b < (setting -> b_size)*(setting -> test_iter); b++) {
        for (int i = 0; i < setting -> v_dim; i++) {
            fprintf(setting -> fp_log, "%f ", y_hat[b][i]);
        }
    }
    fprintf(setting -> fp_log, "\n");

    // テストデータの正解値と当てはめ値から評価値(指標はconfig参照)を計算し表示する // accuracy への対応変更を予定
    double r2 = calc_r2(y_test, y_hat, (setting -> b_size)*(setting -> test_iter), setting -> v_dim);
    fprintf(setting -> fp_log, "The coefficient of determination of test data is %f.\n", r2);

    return r2;
}

void run(config *setting, model *M){
    // 各検証において、訓練時の学習決定係数の推移がグラフに出力され、最終的なテストデータにおける決定係数が列挙される   In each validation, the transition of the training coefficient of determination is output to a graph, and the coefficient of determination in the final test data is enumerated
    if (setting -> cross_val == 1) {   // 交差検証を行う
        // 設計したモデルについて、データを与えて、交差検証法により、学習とテストを両方行う関数     A function that provides data for the designed model and performs both learning and testing by cross-validation
        CRVL(setting, M);
    } else {        // ホールドアウト法により、学習とテストを行う
        learn_set(setting, M);
    }

    if (setting -> fprint_graph == 1) {
        fprintf(setting -> fp_graph, "quit\n");
        pclose(setting -> fp_graph);
    }
}