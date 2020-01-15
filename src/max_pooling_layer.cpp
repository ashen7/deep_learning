/*
 * =====================================================================================
 *
 *       Filename:  max_pooling_layer.cpp
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
#include "max_pooling_layer.h"

#include <memory>
#include <vector>
#include <functional>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"

namespace cnn {

MaxPoolingLayer::MaxPoolingLayer() {
}

MaxPoolingLayer::~MaxPoolingLayer() {
}

/*
 * 初始化卷积层
 * 输入的宽 高 深度 
 * 卷积核的宽 高 深度 个数
 * 补0填充 步长 学习率
 */
void MaxPoolingLayer::Initialize(int input_height, int input_width,
                                 int channel_number, int filter_height,
                                 int filter_width, int stride) {
    if (input_height <= 0
            || input_width <= 0
            || filter_height <= 0
            || filter_width <= 0
            || stride <= 0) {
        LOG(ERROR) << "maxpooling layer initialize failed, input parameter is wrong";
        return ;
    }

    input_height_ = input_height;
    input_width_ = input_width;
    channel_number_ = channel_number;
    filter_height_ = filter_height;
    filter_width_ = filter_width;
    stride_ = stride;
    
    //初始化下采样输出数组 深度(不变) 高 宽
    output_height_ = CalculateOutputHeight(input_height, filter_height, stride);
    output_width_ = CalculateOutputWidth(input_width, filter_width, stride);
    Matrix::CreateZeros(channel_number_, output_height_, output_width_, output_array_);
}

/*
 * 池化层的前向计算
 * 输入是卷积提取的特征图 
 * 根据步长S max pooling的池化核filter大小x * x 在feature map上移动 每个x*x的filter中结果是最大值
 */
int MaxPoolingLayer::Forward(const Matrix3d& input_array) {
    input_array_ = input_array;

    //遍历下采样输出数组深度 得到每个深度的输出
    for (int i = 0; i < channel_number_; i++) {
        if (-1 == Matrix::MaxPoolingForward(input_array_[i], filter_height_, 
                                            filter_width_, stride_, 
                                            output_array_[i])) {
            LOG(ERROR) << "max pooling layer forward failed";
            return -1;
        }
    }
    
    return 0;
}

/*
 * 池化层的反向计算  误差传递从本层到上一层
 * 对于max pooling来说 就是把本层的sensitivity map中每个值返回给对应上一层x*x中最大值的那个坐标 
 * 而上一层其余误差项都为0
 */
int MaxPoolingLayer::Backward(const Matrix3d& input_array, 
                              const Matrix3d& sensitivity_array) {
    //初始化delta array 存放上一层的sensitivity map
    Matrix::CreateZeros(Matrix::GetShape(input_array_), delta_array_);

    //遍历下采样输出数组深度 从输入找出每个最大值的索引 赋值到上一层sensitivity map相应位置
    for (int i = 0; i < channel_number_; i++) {
        if (-1 == Matrix::MaxPoolingBackward(input_array_[i], sensitivity_array[i], 
                                             filter_height_, filter_width_, 
                                             stride_, delta_array_[i])) {
            LOG(ERROR) << "max pooling layer backward failed";
            return -1;
        }
    }

    return 0;
}



}       //namespace cnn

