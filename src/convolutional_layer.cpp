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
#include "filter.h"

#include <memory>
#include <vector>
#include <functional>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"

namespace cnn {

//静态成员的初始化
ConvolutionalLayer::ReLuActivatorCallback ConvolutionalLayer::relu_forward_callback_(nullptr);
ConvolutionalLayer::ReLuActivatorCallback ConvolutionalLayer::relu_backward_callback_(nullptr);

ConvolutionalLayer::ConvolutionalLayer() {
}

ConvolutionalLayer::~ConvolutionalLayer() {
    filters_.clear();
}

/*
 * 初始化卷积层
 * 输入的宽 高 深度 
 * 卷积核的宽 高 深度 个数
 * 补0填充 步长 学习率
 */
void ConvolutionalLayer::Initialize(size_t input_height, size_t input_width, size_t channel_number, 
                                    size_t filter_height, size_t filter_width, size_t filter_number, 
                                    size_t zero_padding, size_t stride, size_t learning_rate) {
    if (0 == input_height
            || 0 == input_width
            || 0 == filter_height
            || 0 == filter_width) {
        LOG(ERROR) << "convolutional layer initialize failed, input parameter is wrong";
        return ;
    }
    input_height_ = input_height;
    input_width_ = input_width;
    channel_number_ = channel_number;
    filter_height_ = filter_height;
    filter_width_ = filter_width;
    filter_number_ = filter_number;
    zero_padding_ = zero_padding;
    stride_ = stride;
    learning_rate_ = learning_rate;
    
    //初始化输出数组特征图 深度(这里是卷积核个数) 高 宽
    size_t output_height = CalculateOutputHeight(input_height, filter_height, zero_padding, stride);
    size_t output_width = CalculateOutputWidth(input_width, filter_width, zero_padding, stride);
    Matrix::CreateZeros(filter_number, output_height, output_width, output_array_);
    
    //初始化每个卷积核
    filters_.reserve(filter_number);
    for (int i = 0; i < filter_number; i++) {
        filters_.push_back(std::make_shared<Filter>());
        //调用每个filter的初始化  初始化filter的权重 和 偏置
        filters_[i]->Initialize(channel_number, filter_height, filter_width);
    }
}

/*
 * 设置每个filter的权重 和 偏置
 */
void ConvolutionalLayer::set_filters(const std::vector<Matrix3d>& filters_weights, 
                                     const std::vector<double> filters_biases) noexcept {
    for (int i = 0; i < filters_weights.size(); i++) {
        filters_[i]->set_weights_array(filters_weights[i]);
    }
    for (int i = 0; i < filters_biases.size(); i++) {
        filters_[i]->set_bias(filters_biases[i]);
    }
}

/*
 * 打印每个filter的权重 和 偏置
 */
void ConvolutionalLayer::Dump() const noexcept {
    //调用每个filter的dump
    for (auto filter : filters_) {
        filter->Dump();
    }
}

}       //namespace cnn

