# Multi-Layer-Perceptron

## Introduction

本稿は、深層ニューラルネットワーク *Deep Neural Network* の基礎的アルゴリズムの学習ないしは研究、あるいは関数近似器としてのDNNの解析への利用を目的とした、*gunnersgoestocl*の制作物である。

極めて限定された一部のライブラリを除き、極力全ての構造や関数を自作した。これは、DNNの基礎的な理解を深めるためである。
-   自由に構造を変更し、新たな機能を追加することが可能となっている。
-   Pythonのライブラリなどではブラックボックス化している、学習時の各ニューロンやネットワークパラメータの挙動を、事細かに追跡することが可能である。
-   Multi-Layer-Perceptron の構造は、自在に設計することが可能であり、これも研究上非常に都合が良いだろう。
-   外部のデータセットを自分で取ってきて、学習させることが可能であり、この際に、説明変数と目的変数をそれぞれ自由に選ぶことが許されている。

ただし、データセットに関しては、いくつか条件がある。
-   データセットの一行目は、各変数の名前が```;```区切りで書かれていること。
-   データセットの二行目以降は、各変数の値が```;```区切りで書かれていること。
-   ```NULL```値が、含まれていないこと。

これらの条件は、Pythonの```pandas```ライブラリを用いて事前にデータセットを加工しておくことなどで、対応していただけると幸いである。

## 内容物

さまざまな機能を提供するために、main関数が含まれる```adv_regression.c```だけでなく、いくつかのファイルに分割している。

### adv_regression

メイン関数を含んでいる。以下の3種類のファイルポインタの生成が可能である。
-   ```FILE *fp``` : データセットの読み込み用
-   ```FILE *fp_log``` : 学習、検証の結果を記録するためのファイルポインタ
-   ```FILE *fp_graph``` : 学習における、モデルの性能を表す決定係数の推移を表すグラフ(pngファイル)のファイルポインタ

現時点で、学習済みパラメータを保存することはできない仕様にしているが、小規模なアップデートにより、実装することが可能である。

その後、変数の指定、ニューラルネットワークの設計、学習の設定、記録の設定を、標準入力により行う。これらは、指示に従って進めれば問題ない。

### setting.c

様々な、変数の設定、関数の設定、学習の設定、記録の設定を行う関数が含まれている。

### init.c

構造体の定義、ニューラルネットワークの初期化、順伝播、誤差逆伝播、ニューロン・ネットワークの格納する値の表示が含まれている。

### learn.c

学習を進めるための関数が含まれている。交差検証の関数、初期化からデータの読み込み、学習、検証、記録などの関数が含まれている。

## 実行方法

```
make
```
により、私が恣意的に生成したほぼ線形回帰可能な小規模データセットを用いて、学習を行うことができる。

以下のように、聞かれた内容に従って入力していくことで、学習を行うことができる。
```
The variables are "x1" "x2" "x3" "x4" "y" 
Input the name of the objective variable: "y"
You input "y".
v_col: , token: "y"
"y" is successfully registered.
v_dim: 1, v_col:4 
Input the name of the objective variables (END to quit selecting): END
You input END.
Input the name of the explanatory variables (ALL to select all or END to quit selecting): "x1"
You input "x1".
v_col: 4 , token: "x1"
"x1" is successfully registered.
o_dim: 1, o_col:0 
Input the name of the explanatory variables (ALL to select all or END to quit selecting): "x2"
You input "x2".
v_col: 4 , token: "x2"
"x2" is successfully registered.
o_dim: 2, o_col:0 1 
Input the name of the explanatory variables (ALL to select all or END to quit selecting): END
You input END.
The column number of the objective variables are 4 .
The column number of the explanatory variables are 0 1 .
Start designing the neural network.
Input the number of middle layers (& put ENTER): 2

Input the number of neurons in each middle layer (put ENTER everytime!):
5
5
The number of  neuron layers is 4.
The number of neurons in each layer are 
    input layer:       2
     1 middle layer:   5
     2 middle layer:   5
    output layer:       1
Input the learning rate: 0.0001
The learning rate is 0.000100.
Now Loading Data File...
The number of data is 19.
Input the batch size: 3
Input the number of iterations: 5
The batch size is 3.
The number of iterations is 5.
So, the size of test data is 3.
Input the total number of epochs: 10
The total number of epochs is 10.
Do you want to apply Cross Validation Method?
    1:yes   2:no    (input 1 or 2)  :2  // 1を押すと、交差検証法を適用する
You can choose the contents to be recorded in the log file.
The contents shown below will be necessarily recorded in the log file.
    1. The objective variable
    2. The explanatory variables
    3. The number of neurons in each layer
    4. The learning rate
    5. The total number of data
    6. The batch size
    7. The number of iterations
    8. The size of test data
    9. The total number of epochs
    10. The coefficient of determination of test data
Then, you can choose the additional contents to be recorded in the log file.
Do you want the confirmation sentences that the program is working properly?
    1:yes   2:no    (input 1 or 2) :
2   // 1を押すと、プログラムが正常に動作しているかの確認文が逐一全て表示される
Do you want the values which neurons and weights take in every stage?
    1:yes   2:no    3:only if iteration ends   (input 1,2 or 3) :
2   // 1を押すと、各ニューロンの値や重みの推移が逐一全て記録される
```
なお、外部からデータセットを読み込む場合には、以下のように入力することで、データセットを読み込むことができる。
```
make
```
その際に、あらかじめ、データセットを```data```ディレクトリに格納し、Makefileを次のように修正しておくことが必要である。
```
sample: bin/demo
	bin/demo data/********* log/sample_output_$(TODAY).txt log/sample_output_$(TODAY)_graph
```

