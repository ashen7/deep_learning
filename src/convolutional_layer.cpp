/*
 * =====================================================================================
 *
 *       Filename:  convolutional_layer.cpp
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
#include "convolutional_layer.h"

#include <memory>
#include <vector>
#include <functional>

#include <glog/logging.h>

#include "utility/matrix.hpp"

namespace dnn {

//静态成员的初始化
//ConvolutionalLayer::SigmoidActivatorCallback ConvolutionLayer::forward_activator_callback_(nullptr);
//ConvolutionLayer::SigmoidActivatorCallback ConvolutionLayer::backward_activator_callback_(nullptr);

ConvolutionalLayer::ConvolutionalLayer() {
}

ConvolutionalLayer::~ConvolutionalLayer() {
    filters.clear();
}

/*
 * 初始化全连接层
 * 初始化权重数组 偏置数组为一个很小的值
 */
void ConvolutionLayer::Initialize(size_t input_node_size, 
                                    size_t output_node_size) {
    input_node_size_ = input_node_size;
    output_node_size_ = output_node_size; 

    //生成随机数 来初始化 权重数组 
    //比如第一层8个节点 第二层10个 权重就是10行8列
    Random::Uniform(-0.1, 0.1, output_node_size, input_node_size, weights_array_);
    Matrix::CreateZerosMatrix(output_node_size, 1, biases_array_);
    Matrix::CreateZerosMatrix(output_node_size, 1, output_array_);
}

