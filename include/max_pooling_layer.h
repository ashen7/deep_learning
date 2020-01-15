/*
 * =====================================================================================
 *
 *       Filename:  max_pooling_layer.h
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
#ifndef CNN_MAX_POOLING_LAYER_H_
#define CNN_MAX_POOLING_LAYER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>
#include <functional>

#include "utility/singleton.hpp"

namespace cnn {

//全连接层实现类
class MaxPoolingLayer {
public:
    typedef std::vector<double> Matrix1d;
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;

    MaxPoolingLayer();
    ~MaxPoolingLayer();
    MaxPoolingLayer(const MaxPoolingLayer&) = delete;
    MaxPoolingLayer& operator =(const MaxPoolingLayer&) = delete;
    MaxPoolingLayer(MaxPoolingLayer&&) = default;
    MaxPoolingLayer& operator =(MaxPoolingLayer&&) = default;
    
    //计算下采样输出的宽高
    //输出大小 = (输入大小 - 池化核大小) / 步长 + 1
    static int CalculateOutputHeight(int input_height, int filter_height, int stride) {
        return static_cast<int>((input_height - filter_height) / stride + 1);
    }

    static int CalculateOutputWidth(int input_width, int filter_width, int stride) {
        return static_cast<int>((input_width - filter_width) / stride + 1);
    }

    const Matrix3d& get_output_array() const noexcept {
        return output_array_;
    }

    const Matrix3d& get_delta_array() const noexcept {
        return delta_array_;
    }

public:
    void Initialize(int input_height, int input_width, 
                    int channel_number, int filter_height, 
                    int filter_width, int stride); 
    int Forward(const Matrix3d& input_array);
    int Backward(const Matrix3d& input_array, 
                 const Matrix3d& sensitivity_array);

private:
    int input_height_;      //输入图像高
    int input_width_;       //输入图像宽
    int channel_number_;    //输入图像深度
    int filter_height_;     //池化核高
    int filter_width_;      //池化核宽
    int stride_;            //池化时移动的步长
    int output_height_;     //下采样输出高
    int output_width_;      //下采样输出宽
    Matrix3d input_array_;  //输入数组
    Matrix3d output_array_; //下采样输出数组
    Matrix3d delta_array_;  //通过误差传递 保存上一层的误差项 
};


}      //namespace cnn

//单例模式
typedef typename utility::Singleton<cnn::MaxPoolingLayer> SingletonPoolLayer;

#endif //CNN_MAX_POOLING_LAYER_H_
