#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    // 入力した文字列を比較する
    if (argc == 2) {
        if (strcmp(argv[1], "\"test\"") == 0) {
            printf("test\n");
        } else {
            printf("not test\n");
        }
    } else {
        printf("引数が足りません\n");
    }
}