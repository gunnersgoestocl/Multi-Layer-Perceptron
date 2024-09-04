#include "setting.h"

// struct config_ {
//     FILE *fp;       // データファイルのポインタ
//     FILE *fp_log;   // ログファイルのポインタ
//     FILE *fp_graph; // グラフファイルのポインタ

//     int v_dim;      // 目的変数の個数(出力層の次元)
//     int *v_col;     // 目的変数の列番号を格納する配列
//     int o_dim;      // 説明変数の個数(入力層の次元)
//     int *o_col;     // 説明変数の列番号を格納する配列

//     int *v_class;       // 出力層の各次元のクラス数 (分類問題の場合)

//     int data_size;      // データの行数
//     int b_size;         // バッチサイズ
//     int iter;           // イテレーション数
//     int test_size;      // テストデータのサイズ
//     int test_iter;      // テストデータのイテレーション数
//     int total_epoch;    // エポック数
//     double alpha;       // 学習率
//     model *MODEL;       // ニューラルネットワークのモデル

//     int task_type;      // タスクの種類 (0: 回帰, 1: 分類)
//     double (*eval_func)(double **, double **, int, int); // 性能評価関数の関数ポインタ (R2, accuracy)
//     int test_train;     // モデルの性能評価とモデルの学習のどちらが目的か (0: 学習, 1: 性能評価)

//     int cross_val;      // クロスバリデーションを行うかのフラグ (0: 行わない;ホールドアウト, 1: 行う)
//     int test_pos;       // ホールドアウト法における、テストデータの位置
//     int program_log;    // プログラムの処理内容を詳細に出力するかのフラグ
//     int show_values;    // 各ステージでのニューロンの値を記録するかのフラグ
//     int fprint_graph;   // グラフをファイルに出力するかのフラグ
// };

// struct model_ {
//     int o_dim;  // 入力層の次元
//     int v_dim;  // 出力層の次元
//     int D;      // ニューラルネットワークの隠れ層の数
//     int *N;     // 各隠れ層のニューロンの数
//     int *v_class; // 出力層の各次元のクラス数 (分類問題の場合)

//     /* 0:linear, 1:ReLU, 2:Leakly_ReLU, 3:Sigmoid, 4:tanh */
//     int *activator; // 各層の活性化関数の種類
//     double (*activator_func)(double); // 各層の活性化関数の関数ポインタの配列
//     double (*activator_grad)(double); // 各層の活性化関数の導関数の関数ポインタの配列
//     int out_activator; // 出力層の活性化関数の種類
//     double (*out_activator_func)(double); // 出力層の活性化関数の関数ポインタ
//     double (*out_activator_grad)(double); // 出力層の活性化関数の導関数の関数ポインタ

//     double **W; // 各層の重み
//     double **B; // 各層のバイアス (線型結合における定数項)
//     int b_size; // バッチサイズ (再学習の際に使用)
// };

/* 設定を格納した構造体を返す関数 */
// task_type, test_train, cross_val, filename_data, filename_log, fprint_graphを設定する
config *set_config(){
    // 設定を格納する構造体のポインタを宣言     Declare a pointer to a structure that stores settings
    config *setting = (config *)malloc(sizeof(config));

    // タイムスタンプを取得する
    char timestamp[80];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M", localtime(&(time_t){time(NULL)}));

    // タスクの種類を行うか選択する
    printf("Select the task type (regression: 0, classification: 1): ");
    scanf("%d", &setting->task_type);
    while (setting->task_type != 0 && setting->task_type != 1) {
        printf("Please input 0 or 1. Retry: ");
        scanf("%d", &setting->task_type);
    }
    if (setting->task_type == 0) {
        setting->eval_func = calc_r2;
    } else {
        setting->eval_func = calc_accuracy;
    }

    // モデルの性能評価(推論)とモデルの学習のどちらが目的か確認する
    printf("Which is your objective? (model learning: 0, model performance evaluation: 1):");
    scanf("%d", &setting->test_train);
    while (setting->test_train != 0 && setting->test_train != 1) {
        printf("Please input 0 or 1. Retry: ");
        scanf("%d", &setting->test_train);
    }

    // クロスバリデーションを行うか確認する
    if (setting->test_train == 1) {
        printf("Do you want to perform cross-validation or hold-out-method? (hold-out: 0, cross-validation: 1): ");
        scanf("%d", &setting->cross_val);
        while (setting->cross_val != 0 && setting->cross_val != 1) {
            printf("Please input 0 or 1. Retry: ");
            scanf("%d", &setting->cross_val);
        }
    } else {
        setting->cross_val = 0;
    }
    int c;
    while ((c = getchar()) != '\n' && c != EOF);

    // 標準入力から分析したいファイル名を読み込む    Read the file name you want to analyze from standard input
    char filename_data[256];
    printf("Input the name of the file you want to analyze: ");
    scanf("%[^\n]%*1[\n]", filename_data);
    FILE *fp = fopen(filename_data, "r");
    if (fp == NULL) {
        printf("Cannot open %s.\n", filename_data);
        exit(1);
    }
    setting->fp = fp;
    char *filename = get_filename(filename_data);

    // ログファイルの名前を決める    Decide the name of the log file
    char filename_log[256];
    strcpy(filename_log, "log/log_");
    strcat(filename_log, filename); strcat(filename_log, timestamp); strcat(filename_log, ".txt");
    FILE *fp_log = fopen(filename_log, "w");
    if (fp_log == NULL) {
        printf("Cannot open %s.\n", filename_log);
        free(setting);
        exit(1);
    }
    setting->fp_log = fp_log;

    printf("Do you want the confirmation sentences that the program is working properly?\n");
    printf("    1:yes   2:no    (input 1 or 2) : ");
    scanf("%d", &(setting->program_log));
    while (setting->program_log != 1 && setting->program_log != 2) {
        printf("Please input 1 or 2. Retry: ");
        scanf("%d", &(setting->program_log));
    }

    printf("Do you want the values which neurons and weights take in every stage?\n");
    printf("    1:yes   2:no    3:only if iteration ends   (input 1,2 or 3) : ");
    scanf("%d", &(setting->show_values));
    while (setting->show_values != 1 && setting->show_values != 2 && setting->show_values != 3) {
        printf("Please input 1, 2 or 3. Retry: ");
        scanf("%d", &(setting->show_values));
    }

    // 学習中のパフォーマンストラッキングのグラフを表示するか確認する
    printf("Do you want to show the graph of the performance tracking during learning? (yes: 1, no: 0): ");
    scanf("%d", &setting->fprint_graph);
    while (setting->fprint_graph != 0 && setting->fprint_graph != 1) {
        printf("Please input 0 or 1. Retry: ");
        scanf("%d", &setting->fprint_graph);
    }
    FILE *fp_graph;
    if (setting->fprint_graph == 1) {
        // グラフファイルの名前を(自動的に)決める    Decide the name of the graph file
        char filename_graph[256];
        strcpy(filename_graph, "log/graph_");
        strcat(filename_graph, filename); strcat(filename_graph, timestamp);
        fp_graph = popen("gnuplot" , "w");
        fprintf(fp_graph, "set terminal png\n");
        fprintf(fp_graph, "set output \"%s.png\"\n", filename_graph);
        fprintf(fp_graph, "set xlabel \"epoch\"\n");
        if (setting->task_type == 0) {
            fprintf(fp_graph, "set ylabel \"R2\"\n");
            fprintf(fp_graph, "set yrange[-300.0:1.0]\n");
        } else {
            fprintf(fp_graph, "set ylabel \"accuracy\"\n");
            fprintf(fp_graph, "set yrange[-0.1:1.1]\n");
        }
        fprintf(fp_graph, "plot");
    }
    setting->fp_graph = fp_graph;
    return setting;
}

/* 用いるデータの設定 */
// データの行数、バッチサイズ、イテレーション数、テストデータのサイズ、テストデータのイテレーション数、エポック数、学習率を設定する
void set_variables(config *setting){
    // <<<<<<<<<<<<<<<<<<<<<<<<変数の指定>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Acquisition of variables
    setting -> v_col = (int *)malloc(sizeof(int));    // 目的変数の列番号を格納する配列   Array to store the column number of the objective variable
    setting -> v_dim = 0;  // 目的変数の個数   Number of objective variables
    setting -> o_dim = 0;  // 説明変数の個数   Number of explanatory variables
    setting -> o_col = (int *)malloc(sizeof(int));    // 説明変数の列番号を格納する配列   Array to store the column number of the explanatory variable

    FILE *fp = setting -> fp;
    FILE *fp_log = setting -> fp_log;

    // <<<<<<<<<<<<<<<<<<<<<<<<変数の取得>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // ファイルの1行目を';'で区切って、変数名を読み込み表示する
    char buf[256];
    int num_var = 0;    // 変数の総数
    fgets(buf, sizeof(buf), fp);
    char *token = strtok(buf, ";");
    printf("The variables are ");
    while (token != NULL) {
        num_var++;
        printf("%s ", token);
        token = strtok(NULL, ";\n");
    }
    printf("\n");
    fflush(stdin);

    // <<<<<<<<<<<<<<<<<目的変数名の取得>>>>>>>>>>>>>>>>>

    // 目的変数の変数名をコマンドラインに書き込むよう指示する
    printf("Input the name of the objective variable: ");
    char objective[256];
    scanf("%[^\n]%*1[\n]", objective);
    // fgets(objective, sizeof(objective), stdin);
    printf("You input %s.\n", objective);
    // exit(1);
    // 目的変数の変数名が含まれる列番号を調べる
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);
    token = strtok(buf, ";\n");

    int col = 0;    // 列番号のカーソル

    while (strcmp(objective, "END") != 0 || setting -> v_dim == 0) {
        // 目的変数の列番号を調べる
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;

        while (token != NULL) {
            // printf("comparing with %s: %d\n", token, strcmp(token, explanatory));

            // ALLと書いた場合(全ての列を目的変数として処理する)
            if (strcmp(objective, "ALL") == 0) {
                if (strcmp(token, objective) != 0) {
                    setting -> v_col = (int *)realloc(setting -> v_col, sizeof(int) * ((setting -> v_dim) + 1));   // 目的変数の列番号を格納する配列のサイズを1増やす
                    (setting -> v_col)[(setting -> v_dim)] = col;
                    // col++;
                    (setting -> v_dim)++;
                }
                // printf("You input ALL.\n");
                // token = strtok(NULL, ";\n");
            }

            // ENDを入力した場合(目的変数の個数が0の場合を除いて、目的変数の入力を終了する)
            else if (strcmp(objective, "END") == 0) {
                // 目的変数の個数が0の場合、目的変数を少なくとも一つ入力するよう指示する
                if (setting -> v_dim == 0) {
                    printf("Please input at least one objective variable.\n");
                    break;
                }
                break;
            }

            // 何かしらの変数名を書いた場合
            else if (strcmp(token, objective) == 0) {
                // 列番号に重複がある場合は重複していると表示する
                printf("v_col: ");
                for (int i = 0; i < setting -> v_dim; i++) {
                    printf("%d ", (setting -> v_col)[i]);
                }
                printf(", token: %s\n", token);
                
                int flag = 0;
                // 登録済みの目的変数と列番号が重複している場合は、重複していると表示する
                for (int i = 0; i < setting -> v_dim; i++) {
                    if ((setting -> v_col)[i] == col) {
                        printf("The column number %d is already registered as objective variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    }
                }
                
                // 目的変数の列番号を格納する配列に、目的変数の列番号を追加する
                if (flag == 0) {
                    // 配列サイズを1増やす
                    if (setting -> v_dim != 0) {
                        setting -> v_col = (int *)realloc(setting -> v_col, sizeof(int) * ((setting -> v_dim) + 1));  // バグの大元の原因(修正済み)
                    }
                    (setting -> v_col)[setting -> v_dim] = col;
                    (setting -> v_dim)++;
                    printf("%s is successfully registered.\n", objective);
                    col = 0;    // カーソルを先頭に戻す
                    
                    printf("v_dim: %d, v_col:", setting -> v_dim);
                    for (int i = 0; i < setting -> v_dim; i++) {
                        printf("%d ", (setting -> v_col)[i]);
                    }
                    printf("\n");
                }
                break;
            }
            // 追加
            else if (col == num_var - 1){
                printf("Cannot find the objective variable. Please input the correct name of the objective variable.\n");
            }
            // 追加終わり
            token = strtok(NULL, ";\n");
            col++;
        }   // while (token != NULL) の終わり(変数名の探索の終わり)

        // 目的変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
        if (setting -> v_dim == 0) {
            printf("Cannot find the objective variable. Please input the correct name of the objective variable.\n");
            // break;
        }
    
        // 目的変数の列番号を昇順にソートする
        for (int i = 0; i < (setting -> v_dim) - 1; i++) {
            for (int j = i + 1; j < setting -> v_dim; j++) {
                if ((setting -> v_col)[i] > (setting -> v_col)[j]) {
                    int tmp = (setting -> v_col)[i];
                    (setting -> v_col)[i] = (setting -> v_col)[j];
                    (setting -> v_col)[j] = tmp;
                }
            }
        }

        // ALL or 目的変数が1つ以上ある中でのENTER の場合、目的変数の入力を終える
        if ((strcmp(objective, "END") == 0 && (setting -> v_dim) != 0)|| strcmp(objective, "ALL") == 0) {
            break;
        }    

        // それ以外の場合は、次の目的変数を入力する
        printf("Input the name of the objective variables (END to quit selecting): ");
        scanf("%[^\n]%*1[\n]", objective);   
        printf("You input %s.\n", objective);
    }   // while (strcmp(objective, "") != 0) の終わり(目的変数の入力の終わり)

    // logファイルに目的変数名を書き込む
    fprintf(fp_log, "objective variable : ");
    // o_colの要素に対応する変数名をfpの1行目から取得して書き込むとともに、リスト"objective_list"にも格納する
    char **objective_list = (char **)malloc(sizeof(char *) * setting -> v_dim);
    for (int i = 0; i < setting -> v_dim; i++) {
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;
        while (token != NULL) {
            if (col == (setting -> v_col)[i]) {
                fprintf(fp_log, "%s ", token);
                objective_list[i] = (char *)malloc(sizeof(char) * strlen(token));
                strcpy(objective_list[i], token);
            }
            token = strtok(NULL, ";\n");
            col++;
        }
    }

    setting -> v_class = NULL;  // 初期化
    // (分類タスクの場合) 目的変数のクラス数を設定する
    if (setting -> task_type == 1) {
        setting -> v_class = (int *)malloc(sizeof(int) * setting -> v_dim);

        // 二値分類か多値分類かを設定する(2値分類での入力量を減らすため)
        int binary;
        printf("Is output binary classification? (no: 0, yes: 1): ");
        scanf("%d", &binary);
        while (binary != 0 && binary != 1) {
            printf("Please input 0 or 1. Retry: ");
            scanf("%d", &binary);
        }
        if (binary == 1) {
            for (int i = 0; i < setting -> v_dim; i++) {
                setting -> v_class[i] = 2;
            }
        } else {
            for (int i = 0; i < setting -> v_dim; i++) {
                printf("Input the number of classes of %s: ", objective_list[i]);
                scanf("%d", &(setting -> v_class[i]));
            }
        }
    }

    // <<<<<<<<<<<<<<<<<<<<説明変数名の取得>>>>>>>>>>>>>>>>>>>>>

    // 説明変数の変数名を一つずつコマンドラインに書き込むよう指示する
    // ALLと書いてEnterを押すと、目的変数以外全ての列を説明変数として処理する
    // 何も書かずにEnterを押すと、説明変数の入力を終了する     
    printf("Input the name of the explanatory variables (ALL to select all or END to quit selecting): ");
    char explanatory[256];
    scanf("%[^\n]%*1[\n]", explanatory);
    printf("You input %s.\n", explanatory);
    
    col = 0;    // 列番号のカーソル
    while (strcmp(explanatory, "END") != 0 || setting -> o_dim == 0) {
        // 説明変数の列番号を調べる
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;

        while (token != NULL) {
            // printf("comparing with %s: %d\n", token, strcmp(token, explanatory));

            // ALLと書いた場合(目的変数以外全ての列を説明変数として処理する) // 要修正
            if (strcmp(explanatory, "ALL") == 0) {
                if (strcmp(token, objective) != 0) {
                    setting -> o_col = (int *)realloc(setting -> o_col, sizeof(int) * ((setting -> o_dim) + 1));   // 説明変数の列番号を格納する配列のサイズを1増やす
                    setting -> o_col[setting -> o_dim] = col;
                    // col++;
                    (setting -> o_dim)++;
                }
            }

            // ENDを入力した場合(説明変数の個数が0の場合を除いて、説明変数の入力を終了する)
            else if (strcmp(explanatory, "END") == 0) {
                // 説明変数の個数が0の場合、説明変数を少なくとも一つ入力するよう指示する
                if (setting -> o_dim == 0) {
                    printf("Please input at least one explanatory variable.\n");
                    break;
                }
                break;
            }

            // 何かしらの変数名を書いた場合
            else if (strcmp(token, explanatory) == 0) {
                // 列番号に重複がある場合は重複していると表示する
                printf("v_col: ");
                for (int i = 0; i < setting -> v_dim; i++) {
                    printf("%d ", (setting -> v_col)[i]);
                }
                printf(", token: %s\n", token);
                int flag = 0;
                // 目的変数と説明変数の列番号が重複している場合は、重複していると表示する
                for (int i = 0; i < setting -> v_dim; i++){
                    if ((setting -> v_col)[i] == col) {
                        printf("The column number %d is already registered as objective variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    }
                }
                if (flag == 1) {
                    break;
                }
                for (int i = 0; i < setting -> o_dim; i++) {
                    if ((setting -> o_col)[i] == col) {
                        printf("The column number %d is already registered as explanatory variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    }
                }
                // 説明変数の列番号を格納する配列に、説明変数の列番号を追加する
                if (flag == 0) {
                    // 配列サイズを1増やす
                    if (setting -> o_dim != 0) {
                        (setting -> o_col) = (int *)realloc(setting -> o_col, sizeof(int) * ((setting -> o_dim) + 1));  // バグの原因
                    }
                    (setting -> o_col)[setting -> o_dim] = col;        // バグの原因が潜んでいた(修正済み)

                    (setting -> o_dim)++;
                    printf("%s is successfully registered.\n", explanatory);
                    
                    col = 0;    // カーソルを先頭に戻す
                    printf("o_dim: %d, o_col:", setting -> o_dim);
                    for (int i = 0; i < setting -> o_dim; i++) {
                        printf("%d ", (setting -> o_col)[i]);
                    }
                    printf("\n");
                }
                break;
            }
            // 追加
            else if (col == num_var - 1){
                printf("Cannot find the explanatory variable. Please input the correct name of the explanatory variable.\n");
            }
            // 追加終わり
            token = strtok(NULL, ";\n");
            col++;
        }   // while (token != NULL) の終わり(変数名の探索の終わり)

        // 説明変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
        if (setting -> o_dim == 0) {
            printf("Cannot find the explanatory variable. Please input the correct name of the explanatory variable.\n");
            // break;
        }
    
        // 説明変数の列番号を昇順にソートする
        for (int i = 0; i < (setting -> o_dim) - 1; i++) {
            for (int j = i + 1; j < setting -> o_dim; j++) {
                if ((setting -> o_col)[i] > (setting -> o_col)[j]) {
                    int tmp = (setting -> o_col)[i];
                    (setting -> o_col)[i] = (setting -> o_col)[j];
                    (setting -> o_col)[j] = tmp;
                }
            }
        }

        // ALL or 説明変数が1つ以上ある中でのENTER の場合、説明変数の入力を終える
        if ((strcmp(explanatory, "END") == 0 && setting -> o_dim != 0)|| strcmp(explanatory, "ALL") == 0) {
            break;
        }    

        // それ以外の場合は、次の説明変数を入力する
        printf("Input the name of the explanatory variables (ALL to select all or END to quit selecting): ");
        scanf("%[^\n]%*1[\n]", explanatory);   
        printf("You input %s.\n", explanatory);
    }   // while (strcmp(explanatory, "") != 0) の終わり(説明変数の入力の終わり)


    // logファイルに説明変数名を書き込む
    fprintf(fp_log, "explanatory variables : ");


    // o_colの要素に対応する変数名をfpの1行目から取得して書き込むとともに、リスト"explanatory_list"にも格納する
    char **explanatory_list = (char **)malloc(sizeof(char *) * (setting -> o_dim));
    for (int i = 0; i < setting -> o_dim; i++) {
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;
        while (token != NULL) {
            if (col == (setting -> o_col)[i]) {
                fprintf(fp_log, "%s ", token);
                explanatory_list[i] = (char *)malloc(sizeof(char) * strlen(token));
                strcpy(explanatory_list[i], token);
            }
            token = strtok(NULL, ";\n");
            col++;
        }
    }

    // 目的変数と説明変数の列番号を表示する
    printf("The column number of the objective variables are ");
    for (int i = 0; i < setting -> v_dim; i++) {
        printf("%d ", (setting -> v_col)[i]);
    }
    printf(".\n");
    printf("The column number of the explanatory variables are ");
    for (int i = 0; i < setting -> o_dim; i++) {
        printf("%d ", (setting -> o_col)[i]);
    }
    printf(".\n");

}

/* モデルの設計 */
// 入力層の次元、出力層の次元、隠れ層の数、各隠れ層のニューロンの数を設定する
model *set_model(config *setting) {
    // モデルの構造を格納する構造体のポインタを宣言
    model *M = (model *)malloc(sizeof(model));
    setting -> MODEL = M;

    M -> v_class = setting -> v_class;  // (分類問題用) 出力層の各次元のクラス数を格納する配列、回帰ではNULL

    // 入力層の次元を設定
    M -> o_dim = setting -> o_dim;
    // 出力層の次元を設定
    M -> v_dim = setting -> v_dim;
    // ニューラルネットワークの層の数
    printf("Input the number of middle layers (& put ENTER): ");
    scanf("%d", &(M -> D));
    // バッチサイズを設定
    M -> b_size = setting -> b_size;

    M -> N = (int *)malloc(sizeof(int) * (M -> D));    // 各層のニューロンの個数を格納する配列
    M -> activator = (int *)malloc(sizeof(int) * (M -> D));    // 各層の活性化関数の種類を格納する配列
    // 各層のニューロンの個数と、各層の活性化関数の種類を設定する
    for (int i = 0; i < M -> D; i++) {
        printf("\nInput the number of neurons in %dth middle layer: ", i+1);
        scanf("%d", &(M -> N[i]));
        printf("Input the type of activator in %dth middle layer\n  (0:linear, 1:ReLU, 2:Leakly_ReLU, 3:Sigmoid, 4:tanh): ", i+1);
        scanf("%d", &(M -> activator[i]));
    }

    // 出力層の各次元のクラス数を設定する (分類タスクの場合)
    int multi_class = 0;
    if (setting -> task_type == 1) {
        // 初期化
        M -> v_class = (int *)malloc(sizeof(int) * (M -> v_dim));
        M -> v_class_sum = 0;
        // 入力
        printf("Input the number of classes in each dimension of the output layer (put ENTER everytime!):\n");
        for (int i = 0; i < M -> v_dim; i++) {
            scanf("%d", &(M -> v_class[i]));
            if (multi_class == 0 && M -> v_class[i] > 2) {
                multi_class = 1;
            }
            M -> v_class_sum += M -> v_class[i];
        }
    }

    // 出力層の活性化関数の種類を設定する
    if (setting -> task_type == 1 && multi_class == 0) {
        printf("The activator in output layer is Softmax.\n");
        M -> out_activator = 10;
    } else {
        printf("Input the type of activator in output layer.\n");
        if (setting -> task_type == 0) {
            printf(" Our recommendation is 0 (linear).\n");
        } else {
            printf(" Our recommendation is 3 (Sigmoid).\n");
        }
        printf(" (0:linear, 1:ReLU, 2:Leakly_ReLU, 3:Sigmoid, 4:tanh): ");
        scanf("%d", &(M -> out_activator));
        while (M -> out_activator < 0 || M -> out_activator > 4) {
            printf("Please input the number between 0 and 4. Retry: ");
            scanf("%d", &(M -> out_activator));
        }
    }

    return M;
}

/* 設計したモデルの出力 */
void print_model(config *setting, model *M){
    printf("Start designing the neural network.\n");

    // ニューラルネットワークの層の数と各層のニューロンの個数を表示する
    printf("The number of  neuron layers is %d.\n", (M -> D)+2);
    printf("The number of neurons in each layer are \n");
    printf("    input layer: %7d\n", M -> o_dim);
    for (int i = 0; i < M -> D; i++) {
        printf("    %2d middle layer: %3d\n", i+1, M -> N[i]);
    }
    printf("    output layer: %7d\n", M -> v_dim);

    fprintf(setting -> fp_log, "Design of NN: %d -", M -> o_dim);
    for (int i = 0; i < M -> D; i++) {
        fprintf(setting -> fp_log, " %d -", M -> N[i]);
    }
    fprintf(setting -> fp_log, " %d\n", M -> v_dim);
}

/* 学習に関するパラメータの設定 */
// 損失関数、学習率、バッチサイズ、イテレーション数、テストデータのサイズ、テストデータのイテレーション数、エポック数を設定する
void set_learning(config *setting){
    FILE *fp = setting -> fp;
    FILE *fp_log = setting -> fp_log;

    // 損失関数の取得
    printf("Input the type of loss function.\n");
    printf(" (0:MSE, 1:BCE(binary cross-entropy), 2:CCE, 3:KLdivergence): ");
    scanf("%d", &(setting -> loss_func));
    while (setting -> loss_func < 0 || setting -> loss_func > 3) {
        printf("Please input the number between 0 and 3. Retry: ");
        scanf("%d", &(setting -> loss_func));
    }
    fprintf(setting -> fp_log, "The type of loss function : %d\n", setting -> loss_func);

    // 学習率の取得
    printf("Input the learning rate: ");
    scanf("%lf", &(setting -> alpha));
    printf("The learning rate is %f.\n", setting -> alpha);
    fprintf(setting -> fp_log, "The learning rate : %f.\n", setting -> alpha);

    char buf[256];  // ファイルの1行を読み込むためのバッファ

    printf("Now Loading Data File...\n");

    // データサイズの取得
    // データの行数を数える
    setting -> data_size = -1; // 1行目は変数名なので、データの行数は1つ少ない
    while (fgets(buf, sizeof(buf), fp) != NULL) {
        (setting -> data_size)++;
    }
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);    // この時点でfpは2行目を指している(データの1行目を読み飛ばした)
    printf("The number of data is %d.\n", setting -> data_size);
    fprintf(fp_log, "The total number of data : %d.\n", setting -> data_size);

    // バッチサイズとイテレーション数をコマンドラインから読み込む
    printf("Input the batch size: ");
    scanf("%d", &(setting -> b_size));
    printf("Input the number of iterations: ");
    scanf("%d", &(setting -> iter));
    while (setting -> b_size * setting -> iter > setting -> data_size) {
        printf("The product of the batch size and the number of iterations is larger than the number of data.\n");
        printf("Please input the batch size and the number of iterations again.\n");
        printf("Input the batch size: ");
        scanf("%d", &(setting -> b_size));
        printf("Input the number of iterations: ");
        scanf("%d", &(setting -> iter));
    }

    printf("The batch size is %d.\n", setting -> b_size);
    fprintf(fp_log, "The batch size : %d.\n", setting -> b_size);

    printf("The number of iterations is %d.\n", setting -> iter);
    fprintf(fp_log, "The number of iterations : %d.\n", setting -> iter);

    // テストデータのサイズと(テスト時の)イテレーション数を計算する
    setting -> test_size = setting -> data_size - (setting -> b_size) * (setting -> iter);   // テストデータのサイズ
    setting -> test_iter = (int) (setting -> test_size / setting -> b_size);  // テストデータのイテレーション数(テストをミニバッチ単位で行っても結果には影響しない)
    printf("So, the size of test data is %d.\n", setting -> b_size * setting -> test_iter);
    fprintf(fp_log, "The size of test data : %d.\n", setting -> b_size * setting -> test_iter);

    // エポック数を取得する
    printf("Input the total number of epochs: ");
    scanf("%d", &(setting -> total_epoch));
    printf("The total number of epochs is %d.\n", setting -> total_epoch);
    fprintf(fp_log, "The total number of epochs : %d.\n", setting -> total_epoch);

    // ホールドアウト法を行う場合の、テストデータの位置を決める
    // テストデータの位置を、0以上、iter以下の整数からランダムに選ぶ
    srand((unsigned)time(NULL));    // 乱数の種を時間で設定
    setting -> test_pos = rand() % ((setting -> iter)+1);

    // グラフの出力準備
    if (setting -> fprint_graph == 1){
        if (setting->cross_val == 1) {
                for (int i = 0; i < (setting->iter)+1; i++) {
                    fprintf(setting -> fp_graph, " '-' with lines title 'Cross Validation #%d',", i+1);
                }
            } else {
                fprintf(setting -> fp_graph, " '-' with lines title 'Learning'");
            }
            fprintf(setting -> fp_graph, "\n");
    }
}

char *get_filename(const char* path) {
    // スラッシュ '/' を最後に見つけ、そのポインタを返す
    const char *slash = strrchr(path, '/');
    
    // スラッシュが見つかった場合は、その次の文字がファイル名の始まり
    // 見つからなかった場合は、パス自体がファイル名
    if (slash) {
        return (char*)(slash + 1);
    } else {
        return (char*)path;
    }
}