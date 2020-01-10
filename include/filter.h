/*
 * =====================================================================================
 *
 *       Filename:  filter.h
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
#ifndef CNN_FILTER_H_
#define CNN_FILTER_H_

#include <vector>

namespace cnn {

//全连接层实现类
class Filter {
public:
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;

    Filter();
    ~Filter();
    Filter(const Filter&) = delete;
    Filter& operator =(const Filter&) = delete;
    Filter(Filter&&) = default;
    Filter& operator =(Filter&&) = default;
    
    //梯度检查时 要小小的改变一下权重 来查看梯度的浮动变化
    Matrix2d& get_weights_array() noexcept {
        return weights_array_;
    }

    int get_bias() const noexcept {
        return bias_;
    }

    const Matrix2d& get_weights_gradient_array() const noexcept {
        return weights_gradient_array_;
    }

public:
    void Initialize(size_t height, size_t width, size_t depth); 
    void UpdateWeights(double learning_rate);
    void Dump() const noexcept;

private:
    Matrix2d weights_array_;          //权重数组(卷积核 共享权重)
    int bias_;                        //偏置
    Matrix2d weights_gradient_array_; //权重梯度数组
    int bias_gradient_;               //偏置梯度
};


}      //namespace cnn

#endif //CNN_FILTER_H_
