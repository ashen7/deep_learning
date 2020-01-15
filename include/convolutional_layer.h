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
    typedef std::vector<double> Matrix1d;
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
    static int CalculateOutputHeight(int input_height, int filter_height,
                                     int zero_padding, int stride) {
        return static_cast<int>((input_height - filter_height + 2 * zero_padding) / stride + 1);
    }

    static int CalculateOutputWidth(int input_width, int filter_width,
                                    int zero_padding, int stride) {
        return static_cast<int>((input_width - filter_width + 2 * zero_padding) / stride + 1);
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

    const Matrix3d& get_output_array() const noexcept {
        return output_array_;
    }

    const Matrix3d& get_delta_array() const noexcept {
        return delta_array_;
    }

public:
    void Initialize(int input_height, int input_width, int channel_number, 
                    int filter_height, int filter_width, int filter_number, 
                    int zero_padding, int stride, double learning_rate); 
    int Forward(const Matrix3d& input_array);
    int Backward(const Matrix3d& input_array, 
                 const Matrix3d& sensitivity_array);
    void UpdateWeights();
    
    int CalcBpGradient(const Matrix3d& sensitivity_array);
    int CalcBpSensitivityMap(const Matrix3d& sensitivity_array);
    int ExpandSensitivityMap(const Matrix3d& input_sensitivity_array, 
                             Matrix3d& output_sensitivity_array);
    void Dump() const noexcept;

    int GradientCheck(const Matrix3d& input_array);

protected:
    static ReLuActivatorCallback relu_forward_callback_;  //激活函数的前向计算
    static ReLuActivatorCallback relu_backward_callback_; //激活函数的反向计算 

private:
    int input_height_;      //输入图像高
    int input_width_;       //输入图像宽
    int channel_number_;    //输入图像深度
    int filter_height_;     //卷积核高
    int filter_width_;      //卷积核宽
    int filter_number_;     //卷积核个数
    int zero_padding_;      //补0填充
    int stride_;            //卷积时移动的步长
    double learning_rate_;  //学习速率
    int output_height_;     //输出特征图高
    int output_width_;      //输出特征图宽
    Matrix3d input_array_;         //输入数组
    Matrix3d padded_input_array_;  //补0填充后的输入数组
    Matrix3d output_array_;        //输出特征图数组
    Matrix3d delta_array_;         //通过本层敏感图计算得到 上一层的敏感图(误差项) 这是保存上一层的误差项 
    std::vector<std::shared_ptr<Filter>> filters_;  //filter对象数组
};


}      //namespace cnn

//单例模式
typedef typename utility::Singleton<cnn::ConvolutionalLayer> SingletonConvLayer;

#endif //CNN_CONVOLUTIONAL_LAYER_H_
