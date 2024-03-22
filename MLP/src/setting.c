#include "setting.h"

void set_variables(FILE *fp, FILE *fp_log, int *v_dim, int *v_col, int *o_dim, int *o_col){
    // <<<<<<<<<<<<<<<<<<<<<<<<変数の取得>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    // ファイルの1行目を';'で区切って、変数名を読み込み表示する
    char buf[256];
    int num_var = 0;    // 変数の総数
    fgets(buf, sizeof(buf), fp);
    char *token = strtok(buf, ";");
    while (token != NULL) {
        num_var++;
        printf("%s\n", token);
        token = strtok(NULL, ";\n");
    }

    // <<<<<<<<<<<<<<<<<目的変数名の取得>>>>>>>>>>>>>>>>>

    // 目的変数の変数名をコマンドラインに書き込むよう指示する
    printf("Input the name of the objective variable: ");
    char objective[256];
    scanf("%[^\n]%*1[\n]", objective);
    printf("You input %s.\n", objective);
    // 目的変数の変数名が含まれる列番号を調べる
    fseek(fp, 0, SEEK_SET);
    fgets(buf, sizeof(buf), fp);
    token = strtok(buf, ";\n");

    int col = 0;    // 列番号のカーソル
    
    // while (token != NULL) {
    //     // printf("comparing with %s: %d\n", token, strcmp(token, objective));
    //     if (strcmp(token, objective) == 0) {
    //         v_col = col;
    //         printf("register success\n");
    //         break;
    //     }
    //     token = strtok(NULL, ";\n");
    //     col++;
    // }
    // // 目的変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
    // while (v_col == -1) {
    //     printf("Cannot find the objective variable. Please input the correct name of the objective variable.\n");
    //     // objectiveの配列を初期化する
    //     int i=0;
    //     while (objective[i] != '\0') {
    //         objective[i] = '\0';
    //         i++;
    //     }
    //     // char objective[256];
    //     scanf("%[^\n]%*1[\n]", objective);
    //     printf("You input %s.\n", objective);
    //     // 目的変数の変数名が含まれる列番号を調べる
    //     fseek(fp, 0, SEEK_SET);
    //     fgets(buf, sizeof(buf), fp);
    //     token = strtok(buf, ";\n");
    //     col = 0;    
    //     while (token != NULL) {
    //         if (strcmp(token, objective) == 0) {
    //             v_col = col;
    //             printf("register success\n");
    //             break;
    //         }
    //         token = strtok(NULL, ";\n");
    //         col++;
    //     }
    // }

    while (strcmp(objective, "END") != 0 || *v_dim == 0) {
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
                    v_col = (int *)realloc(v_col, sizeof(int) * (*v_dim + 1));   // 目的変数の列番号を格納する配列のサイズを1増やす
                    v_col[*v_dim] = col;
                    // col++;
                    v_dim++;
                }
                // printf("You input ALL.\n");
                // token = strtok(NULL, ";\n");
            }

            // ENDを入力した場合(目的変数の個数が0の場合を除いて、目的変数の入力を終了する)
            else if (strcmp(objective, "END") == 0) {
                // 目的変数の個数が0の場合、目的変数を少なくとも一つ入力するよう指示する
                if (v_dim == 0) {
                    printf("Please input at least one objective variable.\n");
                    break;
                }
                break;
            }

            // 何かしらの変数名を書いた場合
            else if (strcmp(token, objective) == 0) {
                // 列番号に重複がある場合は重複していると表示する
                printf("v_col: ");
                for (int i = 0; i < *v_dim; i++) {
                    printf("%d ", v_col[i]);
                }
                printf(", token: %s\n", token);
                
                int flag = 0;
                // 登録済みの目的変数と列番号が重複している場合は、重複していると表示する
                for (int i = 0; i < *v_dim; i++) {
                    if (v_col[i] == col) {
                        printf("The column number %d is already registered as objective variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    }
                }
                
                // 目的変数の列番号を格納する配列に、目的変数の列番号を追加する
                if (flag == 0) {
                    // 配列サイズを1増やす
                    if (v_dim != 0) {
                        v_col = (int *)realloc(v_col, sizeof(int) * (*v_dim + 1));  // バグの大元の原因(修正済み)
                    }
                    v_col[*v_dim] = col;
                    (*v_dim)++;
                    printf("%s is successfully registered.\n", objective);
                    col = 0;    // カーソルを先頭に戻す
                    
                    printf("v_dim: %d, v_col:", *v_dim);
                    for (int i = 0; i < *v_dim; i++) {
                        printf("%d ", v_col[i]);
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

        // 説明変数の列番号が見つからなかったら正しい変数名を入力するよう指示する
        if (*v_dim == 0) {
            printf("Cannot find the objective variable. Please input the correct name of the objective variable.\n");
            // break;
        }
    
        // 目的変数の列番号を昇順にソートする
        for (int i = 0; i < *v_dim - 1; i++) {
            for (int j = i + 1; j < *v_dim; j++) {
                if (v_col[i] > v_col[j]) {
                    int tmp = v_col[i];
                    v_col[i] = v_col[j];
                    v_col[j] = tmp;
                }
            }
        }

        // ALL or 目的変数が1つ以上ある中でのENTER の場合、目的変数の入力を終える
        if ((strcmp(objective, "END") == 0 && *v_dim != 0)|| strcmp(objective, "ALL") == 0) {
            break;
        }    

        // それ以外の場合は、次の目的変数を入力する
        printf("Input the name of the objective variables (END to quit selecting): ");
        scanf("%[^\n]%*1[\n]", objective);   
        printf("You input %s.\n", objective);
    }   // while (strcmp(objective, "") != 0) の終わり(目的変数の入力の終わり)

    // logファイルに目的変数名を書き込む
    fprintf(fp_log, "objective variable : %s.\n", objective);
    // o_colの要素に対応する変数名をfpの1行目から取得して書き込むとともに、リスト"explanatory_list"にも格納する
    char **objective_list = (char **)malloc(sizeof(char *) * *v_dim);
    for (int i = 0; i < *v_dim; i++) {
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;
        while (token != NULL) {
            if (col == v_col[i]) {
                fprintf(fp_log, "%s ", token);
                objective_list[i] = (char *)malloc(sizeof(char) * strlen(token));
                strcpy(objective_list[i], token);
            }
            token = strtok(NULL, ";\n");
            col++;
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
    while (strcmp(explanatory, "END") != 0 || *o_dim == 0) {
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
                    o_col = (int *)realloc(o_col, sizeof(int) * (*o_dim + 1));   // 説明変数の列番号を格納する配列のサイズを1増やす
                    o_col[*o_dim] = col;
                    // col++;
                    (*o_dim)++;
                }
            }

            // ENDを入力した場合(説明変数の個数が0の場合を除いて、説明変数の入力を終了する)
            else if (strcmp(explanatory, "END") == 0) {
                // 説明変数の個数が0の場合、説明変数を少なくとも一つ入力するよう指示する
                if (*o_dim == 0) {
                    printf("Please input at least one explanatory variable.\n");
                    break;
                }
                break;
            }

            // 何かしらの変数名を書いた場合
            else if (strcmp(token, explanatory) == 0) {
                // 列番号に重複がある場合は重複していると表示する
                printf("v_col: ");
                for (int i = 0; i < *v_dim; i++) {
                    printf("%d ", v_col[i]);
                }
                printf(", token: %s\n", token);
                int flag = 0;
                // 目的変数と説明変数の列番号が重複している場合は、重複していると表示する
                for (int i = 0; i < *v_dim; i++){
                    if (v_col[i] == col) {
                        printf("The column number %d is already registered as objective variables.\n", col);
                        flag = 1;
                        col = 0;
                        break;
                    }
                }
                if (flag == 1) {
                    break;
                }
                for (int i = 0; i < *o_dim; i++) {
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
                    if (*o_dim != 0) {
                        o_col = (int *)realloc(o_col, sizeof(int) * (*o_dim + 1));  // バグの原因
                    }
                    o_col[*o_dim] = col;        // バグの原因が潜んでいた(修正済み)

                    (*o_dim)++;
                    printf("%s is successfully registered.\n", explanatory);
                    
                    col = 0;    // カーソルを先頭に戻す
                    printf("o_dim: %d, o_col:", *o_dim);
                    for (int i = 0; i < *o_dim; i++) {
                        printf("%d ", o_col[i]);
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
        if (*o_dim == 0) {
            printf("Cannot find the explanatory variable. Please input the correct name of the explanatory variable.\n");
            // break;
        }
    
        // 説明変数の列番号を昇順にソートする
        for (int i = 0; i < *o_dim - 1; i++) {
            for (int j = i + 1; j < *o_dim; j++) {
                if (o_col[i] > o_col[j]) {
                    int tmp = o_col[i];
                    o_col[i] = o_col[j];
                    o_col[j] = tmp;
                }
            }
        }

        // ALL or 説明変数が1つ以上ある中でのENTER の場合、説明変数の入力を終える
        if ((strcmp(explanatory, "END") == 0 && *o_dim != 0)|| strcmp(explanatory, "ALL") == 0) {
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
    char **explanatory_list = (char **)malloc(sizeof(char *) * *o_dim);
    for (int i = 0; i < *o_dim; i++) {
        fseek(fp, 0, SEEK_SET);
        fgets(buf, sizeof(buf), fp);
        token = strtok(buf, ";\n");
        col = 0;
        while (token != NULL) {
            if (col == o_col[i]) {
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
    for (int i = 0; i < *v_dim; i++) {
        printf("%d ", v_col[i]);
    }
    printf(".\n");
    printf("The column number of the explanatory variables are ");
    for (int i = 0; i < *o_dim; i++) {
        printf("%d ", o_col[i]);
    }
    printf(".\n");
}