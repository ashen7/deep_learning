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
void ConvolutionalLayer::Initialize(int input_height, int input_width, int channel_number, 
                                    int filter_height, int filter_width, int filter_number, 
                                    int zero_padding, int stride, double learning_rate) {
    if (input_height <= 0
            || input_width <= 0
            || filter_height <= 0
            || filter_width <= 0
            || zero_padding < 0
            || stride <= 0) {
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
    output_height_ = CalculateOutputHeight(input_height, filter_height, zero_padding, stride);
    output_width_ = CalculateOutputWidth(input_width, filter_width, zero_padding, stride);
    Matrix::CreateZeros(filter_number_, output_height_, output_width_, output_array_);
    
    //初始化每个卷积核
    filters_.reserve(filter_number);
    for (int i = 0; i < filter_number; i++) {
        filters_.push_back(std::make_shared<Filter>());
        //调用每个filter的初始化  初始化filter的权重 和 偏置
        filters_[i]->Initialize(channel_number, filter_height, filter_width);
    }
   
    //绑定激活函数
    if (nullptr == relu_forward_callback_) {
        set_relu_forward_callback(std::bind(Activator::ReLuForward, 
                                            std::placeholders::_1,
                                            std::placeholders::_2));
    }
    if (nullptr == relu_backward_callback_) {
        set_relu_backward_callback(std::bind(Activator::ReLuBackward, 
                                             std::placeholders::_1,
                                             std::placeholders::_2));
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
 * 卷积层的前向计算
 * 先给输入数组补0填充 
 * 遍历每一个filter 在输入图像上卷积 得到每一个filter提取的特征图
 * 最后经过ReLu激活函数 得到前向计算的输出结果特征图
 */
int ConvolutionalLayer::Forward(const Matrix3d& input_array) {
    //把输入数组存入成员变量 求激活函数导数时要用到  补0填充的也存入 求梯度时用到做为输入 filter是sensitivity map
    input_array_ = input_array;
    if (-1 ==Matrix::ZeroPadding(input_array, zero_padding_, padded_input_array_)) {
        LOG(ERROR) << "convolutional layer forward failed";
        return -1;
    }
    
    //遍历每个filter 在输入图像上进行互相关操作 输出特征图深度和filter个数一致
    for (int i = 0; i < filters_.size(); i++) {
        if (-1 == Matrix::Convolution(padded_input_array_, filters_[i]->get_weights_array(), 
                                      output_array_[i], filters_[i]->get_bias(), stride_)) {
            LOG(ERROR) << "convolutional layer forward failed";
            return -1;
        }
    }
    
    //最后对得到的特征图 经过ReLu激活函数 得到前向计算的输出结果
    if (relu_forward_callback_) {
        relu_forward_callback_(output_array_, output_array_);
    } else {
        LOG(ERROR) << "convolutional layer forward failed, relu activator forward callback function is empty";
        return -1;
    }
    
    return 0;
}

/*
 * 卷积层的反向计算
 * 1. 前向计算输出特征图
 * 2. 反向计算传递给前一层的误差项(敏感图)
 * 3. 利用本层的误差项 计算前一层每个权重的梯度 偏置的梯度
 */
int ConvolutionalLayer::Backward(const Matrix3d& input_array, 
                                 const Matrix3d& sensitivity_array) {
    if (-1 == Forward(input_array)) {
        LOG(ERROR) << "convolutional layer backward failed";
        return -1;
    }
    if (-1 == CalcBpSensitivityMap(sensitivity_array)) {
        LOG(ERROR) << "convolutional layer backward failed";
        return -1;
    }
    if (-1 == CalcBpGradient(sensitivity_array)) {
        LOG(ERROR) << "convolutional layer backward failed";
        return -1;
    }

    return 0;
}

/*
 * 遍历每个filter 利用梯度下降优化算法 来更新网络权重
 */
void ConvolutionalLayer::UpdateWeights() {
    for (auto filter : filters_) {
        filter->UpdateWeights(learning_rate_);
        LOG(INFO) << learning_rate_;
    }
}

/*
 * 计算每个filter的每个权重梯度
 * 1. 用本层sensitivity map作为卷积核filter 先还原成步长为1得到sensitivity map
 * 2. 在input上进行互相关操作 得到的就是filter的权重梯度
 * 3. 偏置项梯度 就是本层sensitivity map所有误差项之和
 */
int ConvolutionalLayer::CalcBpGradient(const Matrix3d& sensitivity_array) {
    //处理卷积步长为S时 对原始sensitivity_array进行扩展
    Matrix3d expanded_sensitivity_array;
    if (-1 == ExpandSensitivityMap(sensitivity_array, expanded_sensitivity_array)) {
        LOG(ERROR) << "calculate bp weights gradient and bias gradient failed";
        return -1;
    }

    //遍历每个filter 计算每个filter的权重梯度 和 偏置梯度
    for (int i = 0; i < filters_.size(); i++) {
        auto filter = filters_[i];
        Matrix3d weights_gradient_array(channel_number_, 
                                        Matrix2d(filter_height_, 
                                        Matrix1d(filter_width_, 0)));
        //遍历每个filter的深度 计算sensitivity map 在input 上的每个通道进行互相关操作
        for (int j = 0; j < channel_number_; j++) {
            if (-1 == Matrix::Convolution(padded_input_array_[j], expanded_sensitivity_array[i], 
                                          weights_gradient_array[j], 0.0, 1)) {
                LOG(ERROR) << "calculate bp weights gradient and bias gradient failed";
                return -1;
            }
        }
        
        //设置权重梯度
        filter->set_weights_gradient_array(weights_gradient_array);
        //bias偏置项梯度就是sensitivity map所有误差项之和
        filter->set_bias_gradient(Matrix::Sum(sensitivity_array[i]));
    }
    
    return 0;
}

/*
 * 计算反向传播误差传递  传递给上一层的误差项(敏感图)
 * 输入是本层的误差项 
 * 1. 将本层sensitivity map还原成步长为1时的sensitivity map
 * 2. 将sensitivity map外圈补0
 * 3. 将每个filter (翻转180度 遍历每个深度 和对应filter的本层sensitivity map进行卷积运算)
 * 4. 上一层的结果sensitivity map就是 每个filter计算出来的上一层sensitivity map之和
 */
int ConvolutionalLayer::CalcBpSensitivityMap(const Matrix3d& sensitivity_array) {
    //处理卷积步长为S时 对原始sensitivity_array进行扩展
    Matrix3d expanded_sensitivity_array;
    if (-1 == ExpandSensitivityMap(sensitivity_array, expanded_sensitivity_array)) {
        LOG(ERROR) << "calculate bp sensitivity map failed";
        return -1;
    }

    auto shape = Matrix::GetShape(expanded_sensitivity_array);
    if (shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "calculate bp sensitivity map failed";
        return -1;
    }

    //对sensitivity map进行补0填充
    int expanded_width = std::get<2>(shape);
    int zero_padding = (input_width_ + filter_width_ - 1 - expanded_width) / 2;
    Matrix3d padding_expanded_sensitivity_array;
    if (-1 == Matrix::ZeroPadding(expanded_sensitivity_array, zero_padding, 
                                  padding_expanded_sensitivity_array)) {
        LOG(ERROR) << "calculate bp sensitivity map failed";
        return -1;
    }

    //这里判断一下delta_array有没有初始化 
    if (!Matrix::MatrixCheck(delta_array_, channel_number_, 
                             input_height_, input_width_, false)) {
        delta_array_.clear();
        delta_array_ = Matrix3d(channel_number_, Matrix2d(input_height_, Matrix1d(input_width_, 0)));
    }
    
    //遍历每个filter 把每个filter得到的上层sensitivity map相加就是上层sensitivity map结果
    for (int i = 0; i < filters_.size(); i++) {
        auto filter = filters_[i];
        //将每个filter的权重翻转180度
        Matrix3d flipped_weights_array;
        if (-1 == Matrix::Flip(filter->get_weights_array(), 
                               flipped_weights_array)) {
            LOG(ERROR) << "calculate bp sensitivity map failed";
            return -1;
        }

        Matrix3d delta_array = Matrix3d(channel_number_, Matrix2d(input_height_, Matrix1d(input_width_, 0)));
        //遍历每个filter的 每个权重深度 
        //一个filter的本层sensitivity map卷积 翻转后filter的每个通道 得到一个filter的上层sensitivity map
        for (int j = 0; j < channel_number_; j++) {
            if (-1 == Matrix::Convolution(padding_expanded_sensitivity_array[i], 
                                          flipped_weights_array[j], delta_array[j], 
                                          0.0, 1)) {
                LOG(ERROR) << "calculate bp sensitivity map failed";
                return -1;
            }
        }

        //将每一个filter的上一层sensitivity map结果加起来
        if (-1 == Matrix::Add(delta_array_, delta_array, delta_array_)) {
            LOG(ERROR) << "calculate bp sensitivity map failed";
            return -1;
        }
    }

    //这里已经得到了上一层的sensitivity map
    //求激活函数的偏导(上一层的加权输入) derivative导数
    Matrix3d derivative_array;
    if (relu_backward_callback_) {
        relu_backward_callback_(input_array_, derivative_array);
    } else {
        LOG(ERROR) << "calculate bp sensitivity map failed, relu backward activator callback is empty";
        return -1;
    }
    
    //最后用上一层sensitivity map 和 输入数组的激活函数偏导 相乘得到上一层sensitivity map结果
    if (-1 == Matrix::HadamarkProduct(delta_array_, derivative_array, delta_array_)) {
        LOG(ERROR) << "calculate bp sensitivity map failed";
        return -1;
    }
    
    return 0;
}

/*
 * 扩展误差项(敏感图) 还原为步长为1时对应的sensitivity map
 */
int ConvolutionalLayer::ExpandSensitivityMap(const Matrix3d& input_sensitivity_array, 
                                             Matrix3d& output_sensitivity_array) {
    if (!Matrix::MatrixCheck(input_sensitivity_array, output_array_, true)) {
        LOG(ERROR) << "expand sensitivity map failed, input sensitivity map shape is wrong";
        return -1;
    }
    
    //得到步长为1时的sensitivity map的深度 高 宽
    int expanded_depth = filter_number_;
    int expanded_height = input_height_ - filter_height_ + 2 * zero_padding_ + 1;
    int expanded_width = input_width_ - filter_width_ + 2 * zero_padding_ + 1;
    //构造新的sensitivity map 原来值赋值到相应位置 其余地方补0
    if (!Matrix::MatrixCheck(output_sensitivity_array, expanded_depth, 
                             expanded_height, expanded_width, false)) {
        Matrix::CreateZeros(expanded_depth, expanded_height, expanded_width, output_sensitivity_array);
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历output的shape 也就是输入敏感图的shape
        for (int i = 0; i < filter_number_; i++) {
            for (int j = 0; j < output_height_; j++) {
                for (int k = 0; k < output_width_; k++) {
                    int row_pos = j * stride_;
                    int col_pos = k * stride_;
                    //步长为S时sensitivity map跳过了步长为1时的那些值 那些值赋值为0
                    output_sensitivity_array[i][row_pos][col_pos] = input_sensitivity_array[i][j][k];
                }
            }
        }
    }

    return 0;
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


/*
 * 卷积层的梯度检查 
 */
int ConvolutionalLayer::GradientCheck(const Matrix3d& input_array) { 
    auto error_function = [](const Matrix3d& array) -> double {
        return Matrix::Sum(array);    
    };

    //前向计算特征图输出
    if (-1 == Forward(input_array)) {
        LOG(ERROR) << "convolutional layer gradient check failed, conv forward failed";
        return -1;
    }

    //初始化sensitivity map为全1数组
    Matrix3d sensitivity_array;
    Matrix::CreateOnes(Matrix::GetShape(output_array_), sensitivity_array);

    //计算当前梯度
    if (-1 == Backward(input_array, sensitivity_array)) {
        LOG(ERROR) << "convolutional layer gradient check failed, conv backward failed";
        return -1;
    }
    
    //检查梯度
    double epsilon = 0.001;
    //取一个filter 遍历深度 行 列
    auto filter = filters_[0];
    auto& weights_array = filter->get_weights_array();
    auto weights_gradient_array = filter->get_weights_gradient_array();
    for (int i = 0; i < channel_number_; i++) {
        for (int j = 0; j < filter_height_; j++) {
            for (int k = 0; k < filter_width_; k++) {
                //一个权重加减一个很小的值 然后再计算前向传播
                weights_array[i][j][k] += epsilon;
                Forward(input_array);
                double error_1 = error_function(output_array_);
                
                weights_array[i][j][k] -= 2 * epsilon;
                Forward(input_array);
                double error_2 = error_function(output_array_);
                
                double expect_gradient = (error_1 - error_2) / (2 * epsilon);
                weights_array[i][j][k] += epsilon;

                LOG(INFO) << "weights(" << i << ", " << j << ", "
                          << k << ": expected -- actual " << expect_gradient
                          << " -- " << weights_gradient_array[i][j][k];
            }
        }
    }


}





}       //namespace cnn

