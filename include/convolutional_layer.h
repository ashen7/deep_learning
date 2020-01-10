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

#include <memory>
#include <vector>
#include <functional>

namespace cnn {

//全连接层实现类
class ConvolutionalLayer {
public:
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;
    typedef std::function<void(const Matrix2d&, Matrix2d&)> ReLuActivatorCallback;

    ConvolutionalLayer();
    ~ConvolutionalLayer();
    ConvolutionalLayer(const ConvolutionalLayer&) = delete;
    ConvolutionalLayer& operator =(const ConvolutionalLayer&) = delete;
    ConvolutionalLayer(ConvolutionalLayer&&) = default;
    ConvolutionalLayer& operator =(ConvolutionalLayer&&) = default;
    
    static void set_forward_activator_callback(SigmoidActivatorCallback forward_callback) {
        forward_activator_callback_ = forward_callback;
    }

    static void set_backward_activator_callback(SigmoidActivatorCallback backward_callback) {
        backward_activator_callback_ = backward_callback;
    }
    
    static SigmoidActivatorCallback get_forward_activator_callback() { 
        return forward_activator_callback_;
    }

    static SigmoidActivatorCallback get_backward_activator_callback() {
        return backward_activator_callback_;
    }

public:
    void Initialize(size_t input_height, size_t input_width, size_t channel_number, 
                    size_t filter_height, size_t filter_width, size_t filter_number, 
                    size_t zero_padding, size_t stride); 
    int Forward(const Matrix2d& input_array);
    int Forward(const ImageMatrix2d& input_array);
    int Backward(const Matrix2d& output_delta_array);
    void UpdateWeights(double learning_rate);
    void Dump() const noexcept;

protected:
    static SigmoidActivatorCallback forward_activator_callback_;  //激活函数的前向计算
    static SigmoidActivatorCallback backward_activator_callback_; //激活函数的反向计算 

private:
    size_t input_height_;   //输入图像高
    size_t input_width_;    //输入图像宽
    size_t channel_number_; //输入图像深度
    size_t filter_height_;  //卷积核高
    size_t filter_width_;   //卷积核宽
    size_t filter_number_;  //卷积核个数
    size_t zero_padding_;   //补0填充
    size_t stride_;         //卷积时移动的步长
    size_t output_height_;  //输出特征图高
    size_t output_width_;   //输出特征图宽
    Matrix2d output_array_; //输出特征图数组
    std::vector<std::shared<Filter>> filters;  //filter对象
};


}      //namespace cnn

#endif //CNN_CONVOLUTIONAL_LAYER_H_
