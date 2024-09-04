#include "activator.h"

activators ACTIVATORS[5]= {linear, relu, leakly_relu, sigmoid, tanh};
activators_grad ACTIVATORS_GRAD[5]= {linear_grad, relu_grad, leakly_relu_grad, sigmoid_grad, tanh_grad};

double linear(double v) {
    return v;
}
double linear_grad(double v) {
    return 1;
}
double relu(double v){
    return v > 0 ? v : 0;
}
double relu_grad(double v){
    return v > 0 ? 1 : 0;
}
double leakly_relu(double v){
    return v > 0 ? v : 0.01 * v;
}
double leakly_relu_grad(double v){
    return v > 0 ? 1 : 0.01;
}
double sigmoid(double v){
    return 1 / (1 + exp(-v));
}
double sigmoid_grad(double o){
    return o * (1 - o);
}
double tanh(double v){
    return tanh(v);
}
double tanh_grad(double o){
    return 1 - o * o;
}

/* 多クラス分類用はinit内部で定義 */