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
    
    void set_forward_activator_callback(SigmoidActivatorCallback forward_callback) {
        forward_activator_callback_ = forward_callback;
    }

    void set_backward_activator_callback(SigmoidActivatorCallback backward_callback) {
        backward_activator_callback_ = backward_callback;
    }
    
    const std::vector<std::vector<float>>& get_output_array() const noexcept {
        return output_array_;
    }

public:
    void Initialize(size_t input_node_size, size_t output_node_size, 
                    SigmoidActivatorCallback forward_callback=nullptr, 
                    SigmoidActivatorCallback backward_callback=nullptr);
    int Forward(const std::vector<std::vector<float>>& input_array);
    int Backward(std::vector<std::vector<float>>& delta_array);

private:
    size_t input_node_size_;     //输入节点
    size_t output_node_size_;    //输出节点
    SigmoidActivatorCallback forward_activator_callback_;  //激活函数的前向计算
    SigmoidActivatorCallback backward_activator_callback_; //激活函数的反向计算 
    std::vector<std::vector<float>> weights_array_;        //权重数组
    std::vector<std::vector<float>> biases_array_;         //偏执数组
    std::vector<std::vector<float>> input_array_;          //输入数组
    std::vector<std::vector<float>> output_array_;         //输出数组
};

FullConnectedLayer::FullConnectedLayer() {}

FullConnectedLayer::~FullConnectedLayer() {
    weights_array_.clear();
    biases_array_.clear();
}

void FullConnectedLayer::Initialize(size_t input_node_size, 
                                    size_t output_node_size, 
                                    SigmoidActivatorCallback forward_callback,  
                                    SigmoidActivatorCallback backward_callback) {
    input_node_size_ = input_node_size;
    output_node_size_ = output_node_size; 
    forward_activator_callback_ = forward_callback;
    backward_activator_callback_ = backward_callback;

    //生成随机数 来初始化 权重数组 
    //比如第一层8个节点 第二层10个 权重就是10行8列
    calculate::random::Uniform(-0.1, 0.1, output_node_size, input_node_size, weights_array_);
    calculate::CreateZerosMatrix(output_node_size, 1, biases_array_);
    calculate::CreateZerosMatrix(output_node_size, 1, output_array_);
    
    LOG(INFO) << "权重数组:";
    calculate::MatrixShow(weights_array_);
}

//前向计算 a = f(w .* x)  输出等于激活函数(权重数组 点积 输入数组)
int FullConnectedLayer::Forward(const std::vector<std::vector<float>>& input_array) {
    //得到本层输入矩阵
    input_array_ = input_array;

    //矩阵相乘
    if (0 != calculate::MatrixMultiply(weights_array_, input_array, output_array_)) {
        return -1;
    }
    //矩阵相加
    if (0 != calculate::MatrixAdd(output_array_, biases_array_, output_array_)) {
        return -1;
    }
    
    //激活函数 得到本层输出矩阵
    if (forward_activator_callback_) {
        forward_activator_callback_(output_array_, output_array_);
    } else {
        LOG(WARNING) << "前向传播激活函数为空...";
        return -1;
    }

    return 0;
}

//反向计算 
int FullConnectedLayer::Backward(const std::vector<std::vector<float>>& output_array,  
                                 std::vector<std::vector<float>>& delta_array) {
    if (backward_activator_callback_) {
        backward_activator_callback_()
    } else {

    }

}


}      //namespace dnn

#endif //DNN_FULL_CONNECTED_LAYER_H_
