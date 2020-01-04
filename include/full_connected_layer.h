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

namespace dnn {

//全连接层实现类
class FullConnectedLayer {
public:
    typedef std::vector<std::vector<float>> Matrix2d;
    typedef std::vector<std::vector<std::vector<float>>> Matrix3d;
    typedef std::function<void(const Matrix2d&, Matrix2d&)> SigmoidActivatorCallback;
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
    int Forward(const Matrix2d& input_array);
    int Backward(const Matrix2d& output_delta_array);
    void UpdateWeights(float learning_rate);
    void Dump() const noexcept;

protected:
    static SigmoidActivatorCallback forward_activator_callback_;  //激活函数的前向计算
    static SigmoidActivatorCallback backward_activator_callback_; //激活函数的反向计算 

private:
    size_t input_node_size_;          //输入节点
    size_t output_node_size_;         //输出节点
    Matrix2d weights_array_;          //权重数组
    Matrix2d biases_array_;           //偏执数组
    Matrix2d input_array_;            //输入数组
    Matrix2d output_array_;           //输出数组
    Matrix2d delta_array_;            //误差数组
    Matrix2d weights_gradient_array_; //权重梯度数组
    Matrix2d biases_gradient_array_;  //偏置梯度数组
};


}      //namespace dnn

#endif //DNN_FULL_CONNECTED_LAYER_H_
