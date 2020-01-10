/*
 * =====================================================================================
 *
 *       Filename:  convolutional_layer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年12月29日 19时38分17秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef CNN_CONVOLUTIONAL_LAYER_H_
#define CNN_CONVOLUTIONAL_LAYER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>
#include <functional>

#include "utility/singleton.hpp"

namespace cnn {
//类的前置声明
class Filter;

//全连接层实现类
class ConvolutionalLayer {
public:
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;
    typedef std::function<void(const Matrix3d&, Matrix3d&)> ReLuActivatorCallback;

    ConvolutionalLayer();
    ~ConvolutionalLayer();
    ConvolutionalLayer(const ConvolutionalLayer&) = delete;
    ConvolutionalLayer& operator =(const ConvolutionalLayer&) = delete;
    ConvolutionalLayer(ConvolutionalLayer&&) = default;
    ConvolutionalLayer& operator =(ConvolutionalLayer&&) = default;
    
    //计算输出特征图的宽高
    //特征图大小 = (输入大小 - 卷积核大小 + 2*补0填充) / 步长 + 1
    static size_t CalculateOutputHeight(size_t input_height, size_t filter_height,
                                        size_t zero_padding, size_t stride) {
        return static_cast<size_t>((input_height - filter_height + 2 * zero_padding) / stride + 1);
    }

    static size_t CalculateOutputWidth(size_t input_width, size_t filter_width,
                                       size_t zero_padding, size_t stride) {
        return static_cast<size_t>((input_width - filter_width + 2 * zero_padding) / stride + 1);
    }

    static void set_relu_forward_callback(ReLuActivatorCallback forward_callback) {
        relu_forward_callback_ = forward_callback;
    }

    static void set_relu_backward_callback(ReLuActivatorCallback backward_callback) {
        relu_backward_callback_ = backward_callback;
    }
    
    static ReLuActivatorCallback get_forward_activator_callback() { 
        return relu_forward_callback_;
    }

    static ReLuActivatorCallback get_backward_activator_callback() {
        return relu_backward_callback_;
    }

    void set_filters(const std::vector<Matrix3d>& filters_weights, 
                     const std::vector<double> filters_biases) noexcept;

public:
    void Initialize(size_t input_height, size_t input_width, size_t channel_number, 
                    size_t filter_height, size_t filter_width, size_t filter_number, 
                    size_t zero_padding, size_t stride, size_t learning_rate); 
    int Forward(const Matrix2d& input_array);
    int Forward(const ImageMatrix2d& input_array);
    int Backward(const Matrix2d& output_delta_array);
    void UpdateWeights(double learning_rate);
    void Dump() const noexcept;

protected:
    static ReLuActivatorCallback relu_forward_callback_;  //激活函数的前向计算
    static ReLuActivatorCallback relu_backward_callback_; //激活函数的反向计算 

private:
    size_t input_height_;   //输入图像高
    size_t input_width_;    //输入图像宽
    size_t channel_number_; //输入图像深度
    size_t filter_height_;  //卷积核高
    size_t filter_width_;   //卷积核宽
    size_t filter_number_;  //卷积核个数
    size_t zero_padding_;   //补0填充
    size_t stride_;         //卷积时移动的步长
    double learning_rate_;  //学习速率
    size_t output_height_;  //输出特征图高
    size_t output_width_;   //输出特征图宽
    Matrix3d output_array_; //输出特征图数组
    std::vector<std::shared_ptr<Filter>> filters_;  //filter对象数组
};


}      //namespace cnn

//单例模式
typedef typename utility::Singleton<cnn::ConvolutionalLayer> SingletonConvLayer;

#endif //CNN_CONVOLUTIONAL_LAYER_H_
