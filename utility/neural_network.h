/*
 * =====================================================================================
 *
 *       Filename:  neural_network.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年12月29日 21时51分39秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef DNN_NEURAL_NETWORK_H_
#define DNN_NEURAL_NETWORK_H_

#include <vector>
#include <memory>

#include "full_connected_layer.h"

namespace dnn {
//神经网络类
class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

public:
    void Initialize(const std::vector<size_t>& fc_layer_nodes_array);
    void Predict(const std::vector<std::vector<float>>& input_array, 
                 std::vector<std::vector<float>>& output_array);

private:
    std::vector<std::shared_ptr<FullConnectedLayer>> fc_layers_array_;
};

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {
    fc_layers_array_.clear();
}

void NeuralNetwork::Initialize(const std::vector<size_t>& fc_layer_nodes_array) {
    //有n层节点 就构造n-1层fc layer
    int fc_layers_array_size = fc_layer_nodes_array.size() - 1;
    fc_layers_array_.reserve(fc_layers_array_size);

    //遍历 初始化全连接层
    for (int i = 0; i < fc_layers_array_size; i++) {
        fc_layers_array_.push_back(std::make_shared<FullConnectedLayer>());
        //初始化
        fc_layers_array_[i]->Initialize(fc_layer_nodes_array[i], 
                                        fc_layer_nodes_array[i + 1]);

        //设置全连接层的前向计算激活函数回调
        fc_layers_array_[i]->set_forward_activator_callback(std::bind([](const std::vector<std::vector<float>>& input_array, 
                                                            std::vector<std::vector<float>>& output_array) {
            for (int i = 0; i < input_array.size(); i++) {
                for (int j = 0; j < input_array[i].size(); j++) {
                    //exp返回e的x次方 得到0. 1. 2.值 加上1都大于1了 然后用1除  最后都小于1
                    output_array[i][j] = 1.0 / (1.0 + exp(-input_array[i][j])); 
                }
            }
        }, std::placeholders::_1, std::placeholders::_2));

        //设置全连接层的反向计算激活函数回调 得到输出层的误差项
        fc_layers_array_[i]->set_backward_activator_callback(std::bind([](const std::vector<std::vector<float>>& output_array, 
                                                             std::vector<std::vector<float>>& delta_array) {
            for (int i = 0; i < output_array.size(); i++) {
                for (int j = 0; j < output_array[i].size(); j++) {
                    delta_array[i][j] = (output_array[i][j] * (1 - output_array[i][j]));
                }
            }
        }, std::placeholders::_1, std::placeholders::_2));
    }
}

//实现预测 也就是利用当前网络的权值计算节点的输出值
void NeuralNetwork::Predict(const std::vector<std::vector<float>>& input_array, 
                            std::vector<std::vector<float>>& output_array) {
    for (int i = 0; i < fc_layers_array_.size(); i++) {
        if (0 == i) {
            fc_layers_array_[i]->Forward(input_array);
        } else {
            fc_layers_array_[i]->Forward(fc_layers_array_[i - 1]->get_output_array());
        }
        
        if ((i + 1) == fc_layers_array_.size()) {
            output_array = fc_layers_array_[i]->get_output_array();
        }
    }
}



}      //namespace dnn

#endif //DNN_NEURAL_NETWORK_H_

