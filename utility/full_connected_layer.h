/*
 * =====================================================================================
 *
 *       Filename:  full_connected_layer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年12月29日 19时38分17秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef DNN_FULL_CONNECTED_LAYER_H_
#define DNN_FULL_CONNECTED_LAYER_H_

#include <memory>
#include <vector>
#include <functional>

#include "matrix_math_function.hpp"

namespace dnn {

//全连接层实现类
class FullConnectedLayer {
public:
    typedef std::function<void(const std::vector<std::vector<float>>&, 
                               std::vector<std::vector<float>>&)> SigmoidActivatorCallback;
    FullConnectedLayer();
    ~FullConnectedLayer();
    FullConnectedLayer(const FullConnectedLayer&) = delete;
    FullConnectedLayer& operator =(const FullConnectedLayer&) = delete;
    FullConnectedLayer(FullConnectedLayer&&) = default;
    FullConnectedLayer& operator =(FullConnectedLayer&&) = default;
    
    const std::vector<std::vector<float>>& get_output_array() const noexcept {
        return output_array_;
    }

    //优化 复制省略
    const std::vector<std::vector<float>>& get_delta_array() const noexcept {
        return delta_array_;
    }

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
    void Initialize(size_t input_node_size, size_t output_node_size); 
    int Forward(const std::vector<std::vector<float>>& input_array);
    int Backward(const std::vector<std::vector<float>>& output_delta_array);
    void UpdateWeights(float learning_rate);
    void Dump() const noexcept;

protected:
    static SigmoidActivatorCallback forward_activator_callback_;  //激活函数的前向计算
    static SigmoidActivatorCallback backward_activator_callback_; //激活函数的反向计算 

private:
    size_t input_node_size_;     //输入节点
    size_t output_node_size_;    //输出节点
    std::vector<std::vector<float>> weights_array_;          //权重数组
    std::vector<std::vector<float>> biases_array_;           //偏执数组
    std::vector<std::vector<float>> input_array_;            //输入数组
    std::vector<std::vector<float>> output_array_;           //输出数组
    std::vector<std::vector<float>> delta_array_;            //误差数组
    std::vector<std::vector<float>> weights_gradient_array_; //权重梯度数组
    std::vector<std::vector<float>> biases_gradient_array_;  //偏置梯度数组
};

//静态成员的初始化
FullConnectedLayer::SigmoidActivatorCallback FullConnectedLayer::forward_activator_callback_(nullptr);
FullConnectedLayer::SigmoidActivatorCallback FullConnectedLayer::backward_activator_callback_(nullptr);

FullConnectedLayer::FullConnectedLayer() {}

FullConnectedLayer::~FullConnectedLayer() {
    weights_array_.clear();
    biases_array_.clear();
}

void FullConnectedLayer::Initialize(size_t input_node_size, 
                                    size_t output_node_size) {
    input_node_size_ = input_node_size;
    output_node_size_ = output_node_size; 

    //生成随机数 来初始化 权重数组 
    //比如第一层8个节点 第二层10个 权重就是10行8列
    calculate::random::Uniform(-0.1, 0.1, output_node_size, input_node_size, weights_array_);
    calculate::matrix::CreateZerosMatrix(output_node_size, 1, biases_array_);
    calculate::matrix::CreateZerosMatrix(output_node_size, 1, output_array_);
    
    LOG(INFO) << "初始化权重数组:";
    calculate::matrix::MatrixShow(weights_array_);
}

/*
 * 前向计算 a = f(w .* x + b)  输出等于激活函数(权重数组 点积 输入数组 最后数组和偏置数组相加)
 * 下一层前向计算的输入数组 就是上一层的输出数组
 */  
int FullConnectedLayer::Forward(const std::vector<std::vector<float>>& input_array) {
    //得到本层输入矩阵 也就是本层的节点值
    input_array_ = input_array;

    //矩阵相乘  w .* x 得到输出数组
    if (0 != calculate::matrix::MatrixMultiply(weights_array_, input_array, output_array_)) {
        return -1;
    }
    //矩阵相加 w .* x + b
    if (0 != calculate::matrix::MatrixAdd(output_array_, biases_array_, output_array_)) {
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
int FullConnectedLayer::Backward(const std::vector<std::vector<float>>& output_delta_array) {
    std::vector<std::vector<float>> temp_array1;
    if (backward_activator_callback_) {
        // 计算x * (1 - x)
        backward_activator_callback_(input_array_, temp_array1);
    } else {
        LOG(WARNING) << "反向传播激活函数为空...";
        return -1;
    }

    //计算w的转置矩阵 WT 
    std::vector<std::vector<float>> weights_transpose_array;
    if (0 != calculate::matrix::TransposeMatrix(weights_array_, weights_transpose_array)) {
        return -1;
    }
    
    std::vector<std::vector<float>> temp_array2;
    //计算WT .* delta_array
    if (0 != calculate::matrix::MatrixMultiply(weights_transpose_array, output_delta_array, temp_array2)) {
        return -1;
    }


    //计算x * (1 - x) * WT .* delta_array 得到本层的delta_array
    if (0 != calculate::matrix::MatrixHadamarkProduct(temp_array1, temp_array2, delta_array_)) {
        return -1;
    }
    
    //利用上一层的误差项delta_array 计算weights的梯度 delta_array .* xT
    std::vector<std::vector<float>> input_transpose_array;
    if (0 != calculate::matrix::TransposeMatrix(input_array_, input_transpose_array)) {
        return -1;
    }
    
    if (0 != calculate::matrix::MatrixMultiply(output_delta_array, input_transpose_array, weights_gradient_array_)) {
        return -1;
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
void FullConnectedLayer::UpdateWeights(float learning_rate) {
    //权重的变化数组 
    std::vector<std::vector<float>> weights_delta_array;
    calculate::matrix::MatrixMulValue(weights_gradient_array_, learning_rate, weights_delta_array);
    calculate::matrix::MatrixAdd(weights_array_, weights_delta_array, weights_array_);

    //偏置的变化数组
    std::vector<std::vector<float>> biases_delta_array;
    calculate::matrix::MatrixMulValue(biases_gradient_array_, learning_rate, biases_delta_array);
    calculate::matrix::MatrixAdd(biases_array_, biases_delta_array, biases_array_);
    LOG(WARNING) << "梯度数组:";
    calculate::matrix::MatrixShow(weights_gradient_array_);
    LOG(WARNING) << "权重数组:";
    calculate::matrix::MatrixShow(weights_array_);
}

void FullConnectedLayer::Dump() const noexcept {
    LOG(WARNING) << "权重数组:";
    calculate::matrix::MatrixShow(weights_array_); 
}

}      //namespace dnn

#endif //DNN_FULL_CONNECTED_LAYER_H_
