/*
 * =====================================================================================
 *
 *       Filename:  full_connected_layer.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月04日 10时44分21秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "full_connected_layer.h"

#include <memory>
#include <vector>
#include <functional>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"

namespace dnn {

//静态成员的初始化
FullConnectedLayer::SigmoidActivatorCallback FullConnectedLayer::forward_activator_callback_(nullptr);
FullConnectedLayer::SigmoidActivatorCallback FullConnectedLayer::backward_activator_callback_(nullptr);

FullConnectedLayer::FullConnectedLayer() {
}

FullConnectedLayer::~FullConnectedLayer() {
}

/*
 * 初始化全连接层
 * 初始化权重数组 偏置数组为一个很小的值
 */
void FullConnectedLayer::Initialize(size_t input_node_size, 
                                    size_t output_node_size) {
    input_node_size_ = input_node_size;
    output_node_size_ = output_node_size; 

    //生成随机数 来初始化 权重数组 
    //比如第一层8个节点 第二层10个 权重就是10行8列
    Random::Uniform(-0.1, 0.1, output_node_size, input_node_size, weights_array_);
    Matrix::CreateZeros(output_node_size, 1, biases_array_);
    Matrix::CreateZeros(output_node_size, 1, output_array_);
}

/*
 * 前向计算 a = f(w .* x + b)  输出等于激活函数(权重数组 点积 输入数组 最后数组和偏置数组相加)
 * 下一层前向计算的输入数组 就是上一层的输出数组
 */  
int FullConnectedLayer::Forward(const Matrix2d& input_array) {
    //得到本层输入矩阵 也就是本层的节点值
    hidden_input_array_ = input_array;

    //矩阵相乘  w .* x 得到输出数组
    if (-1 == Matrix::DotProduct(weights_array_, input_array, output_array_)) {
        return -1;
    }
    //矩阵相加 w .* x + b
    if (-1 == Matrix::Add(output_array_, biases_array_, output_array_)) {
        return -1;
    }
    
    //激活函数 得到本层输出数组 f(w .* x + b)
    if (forward_activator_callback_) {
        forward_activator_callback_(output_array_, output_array_);
    } else {
        LOG(WARNING) << "前向传播激活函数为空...";
        return -1;
    }

    return 0;
}

/*
 * 前向计算 a = f(w .* x + b)  输出等于激活函数(权重数组 点积 输入数组 最后数组和偏置数组相加)
 * 下一层前向计算的输入数组 就是上一层的输出数组
 */  
int FullConnectedLayer::Forward(const ImageMatrix2d& input_array) {
    //得到本层输入矩阵 也就是本层的节点值
    input_array_ = input_array;
    
    //矩阵相乘  w .* x 得到输出数组
    if (-1 == Matrix::DotProduct(weights_array_, input_array, output_array_)) {
        return -1;
    }
    //矩阵相加 w .* x + b
    if (-1 == Matrix::Add(output_array_, biases_array_, output_array_)) {
        return -1;
    }
    
    //激活函数 得到本层输出数组 f(w .* x + b)
    if (forward_activator_callback_) {
        forward_activator_callback_(output_array_, output_array_);
    } else {
        LOG(WARNING) << "前向传播激活函数为空...";
        return -1;
    }

    return 0;
}

/*
 * 反向计算 x是本层节点的值 WT是权重数组的转置矩阵 .*是点积 delta_array是下一层的误差数组
 * 本层的误差项 = x * (1 - x) * WT .* delta_array
 * w权重的梯度 就是 delta_array .* xT  下一层的误差项 点积 本层节点值的转置矩阵
 * b偏置的梯度 就是 delta_array
 */
int FullConnectedLayer::Backward(const Matrix2d& output_delta_array) {
    Matrix2d temp_array1;
    if (backward_activator_callback_) {
        // 计算x * (1 - x)
        //判断一下是隐藏层还是输入层
        if (is_input_layer_) {
            ImageActivator::SigmoidBackward(input_array_, temp_array1);
        } else {
            backward_activator_callback_(hidden_input_array_, temp_array1);
        }
    } else {
        LOG(WARNING) << "反向传播激活函数为空...";
        return -1;
    }

    //计算w的转置矩阵 WT 
    Matrix2d weights_transpose_array;
    if (-1 == Matrix::Transpose(weights_array_, weights_transpose_array)) {
        return -1;
    }
    
    Matrix2d temp_array2;
    //计算WT .* delta_array
    if (-1 == Matrix::DotProduct(weights_transpose_array, output_delta_array, temp_array2)) {
        return -1;
    }

    //计算x * (1 - x) * WT .* delta_array 得到本层的delta_array
    if (-1 == Matrix::HadamarkProduct(temp_array1, temp_array2, delta_array_)) {
        return -1;
    }
    
    //利用上一层的误差项delta_array 计算weights的梯度 delta_array .* xT
    //这里做下判断 因为输入矩阵有可能是uint8_t类型的
    if (is_input_layer_) {
        ImageMatrix2d input_transpose_array;

        if (-1 == ImageMatrix::Transpose(input_array_, input_transpose_array)) {
            return -1;
        }
        if (-1 == Matrix::DotProduct(output_delta_array, input_transpose_array, weights_gradient_array_)) {
            return -1;
        }
    } else {
        Matrix2d _input_transpose_array;
        
        if (-1 == Matrix::Transpose(hidden_input_array_, _input_transpose_array)) {
            return -1;
        }
        if (-1 == Matrix::DotProduct(output_delta_array, _input_transpose_array, weights_gradient_array_)) {
            return -1;
        }
    }
    

    //利用上一层的误差项delta_array 计算biases的梯度 delta_array
    biases_gradient_array_ = output_delta_array;
    
    return 0;
}

/*
 * 利用梯度下降优化算法(就是让值朝着梯度的反方向走) 更新权重 
 * w = w + learning_rate * w_gradient
 * b = b + learning_rate * b_gradient
 */
void FullConnectedLayer::UpdateWeights(double learning_rate) {
    //权重的变化数组 
    Matrix2d weights_delta_array;
    Matrix::ValueMulMatrix(learning_rate, weights_gradient_array_, weights_delta_array);
    Matrix::Add(weights_array_, weights_delta_array, weights_array_);

    //偏置的变化数组
    Matrix2d biases_delta_array;
    Matrix::ValueMulMatrix(learning_rate, biases_gradient_array_, biases_delta_array);
    Matrix::Add(biases_array_, biases_delta_array, biases_array_);
}

void FullConnectedLayer::Dump() const noexcept {
    LOG(INFO) << "权重数组:";
    Matrix::MatrixShow(weights_array_); 
    LOG(INFO) << "偏置数组:";
    Matrix::MatrixShow(biases_array_);
}

}       //namespace dnn
