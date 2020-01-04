/*
 * =====================================================================================
 *
 *       Filename:  neural_network.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月04日 10时47分21秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "neural_network.h"
#include "full_connected_layer.h"

#include "math.h"

#include <vector>
#include <memory>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"

namespace dnn {
NeuralNetwork::NeuralNetwork() {
}

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
    }

    //设置全连接层的前向计算激活函数回调
    if (nullptr == FullConnectedLayer::get_forward_activator_callback()) {
        FullConnectedLayer::set_forward_activator_callback(
                std::bind([](const std::vector<std::vector<float>>& input_array, 
                std::vector<std::vector<float>>& output_array) {
            //如果输出数组未初始化 
            if (0 == output_array.size()) {
                output_array = input_array;
            }

            //计算 1 / (1 + exp(-input_array))
            for (int i = 0; i < input_array.size(); i++) {
                for (int j = 0; j < input_array[i].size(); j++) {
                    //exp返回e的x次方 得到0. 1. 2.值 加上1都大于1了 然后用1除  最后都小于1
                    output_array[i][j] = 1.0 / (1.0 + exp(-input_array[i][j])); 
                }
            }
        }, std::placeholders::_1, std::placeholders::_2));
    }

    //设置全连接层的反向计算激活函数回调 得到输出层的误差项
    if (nullptr == FullConnectedLayer::get_backward_activator_callback()) {
        FullConnectedLayer::set_backward_activator_callback(
                std::bind([](const std::vector<std::vector<float>>& output_array, 
                std::vector<std::vector<float>>& delta_array) {
            //如果输出数组未初始化 
            if (0 == delta_array.size()) {
                delta_array = output_array;
            }

            //计算 output(1 - output)
            for (int i = 0; i < output_array.size(); i++) {
                for (int j = 0; j < output_array[i].size(); j++) {
                    delta_array[i][j] = output_array[i][j] * (1.0 - output_array[i][j]);
                }
            }
        }, std::placeholders::_1, std::placeholders::_2));
    }
}

/*
 * 训练网络  
 * 训练集  标签
 * 迭代轮数 和 学习率
 */
int NeuralNetwork::Train(const std::vector<std::vector<std::vector<float>>>& training_data_set, 
                         const std::vector<std::vector<std::vector<float>>>& labels,
                         int epoch, float learning_rate) {
    //迭代轮数
    for (int i = 0; i < epoch; i++) {
        //遍历每一个输入特征 拿去训练 训练完所有数据集 就是训练完成一轮
        for (int d = 0; d < training_data_set.size(); d++) {
            if (0 != TrainOneSample(training_data_set[d], labels[d], learning_rate)) {
                LOG(ERROR) << "训练失败...";
                return -1;
            }
        }
    }
}

/*
 * 内部函数 训练一个样本(输入特征)x 
 * Predict 前向计算 计算网络节点的输出值
 * CalcGradient 反向计算 从输出层开始往前计算每层的误差项 和权重梯度 偏置梯度
 * UpdateWeights 得到了梯度 利用梯度下降优化算法 更新权重和偏置
 */
int NeuralNetwork::TrainOneSample(const std::vector<std::vector<float>>& input_array, 
                                  const std::vector<std::vector<float>>& label, 
                                  float learning_rate) {
    std::vector<std::vector<float>> output_array;
    if (0 != Predict(input_array, output_array)) {
        return -1;
    }

    if (0 != CalcGradient(output_array, label)) {
        return -1;
    }

    UpdateWeights(learning_rate);

    return 0;
}

/* 
 * 前向计算 实现预测 也就是利用当前网络的权值计算节点的输出值 
 */
int NeuralNetwork::Predict(const std::vector<std::vector<float>>& input_array, 
                           std::vector<std::vector<float>>& output_array) {
    for (int i = 0; i < fc_layers_array_.size(); i++) {
        if (0 == i) {
            if (0 != fc_layers_array_[i]->Forward(input_array)) {
                return -1;
            }
        } else {
            if (0 != fc_layers_array_[i]->Forward(fc_layers_array_[i - 1]->get_output_array())) {
                return -1;
            }
        }
        
        if ((i + 1) == fc_layers_array_.size()) {
            output_array = fc_layers_array_[i]->get_output_array();
        }
    }
}

/*
 * 反向计算 计算误差项和梯度 
 * 节点是输出层是 输出节点误差项delta=output(1 - output)(label - output)
 * 通过输出层的delta 从输出层反向计算 依次得到前面每层的误差项 
 * 得到误差项再计算梯度 更新权重使用
 */
int NeuralNetwork::CalcGradient(const std::vector<std::vector<float>>& output_array, 
                                const std::vector<std::vector<float>>& label) {
    //得到输出节点output
    //const auto& output_array = fc_layers_array_[fc_layers_array_.size() - 1]->get_output_array();
    
    //得到output(1 - output)
    std::vector<std::vector<float>> delta_array; 
    if (FullConnectedLayer::get_backward_activator_callback()) {
        auto backward_activator_callback = FullConnectedLayer::get_backward_activator_callback();
        backward_activator_callback(output_array, delta_array);
    } else {
        LOG(WARNING) << "反向传播激活函数为空...";
        return -1;
    }
     
    //计算(label - output)
    std::vector<std::vector<float>> sub_array; 
    if (0 != calculate::matrix::MatrixSubtract(label, output_array, sub_array)) {
        return -1;
    }

    //再计算output(1 - output)(label - output)  得到输出层的delta array误差项
    if (0 != calculate::matrix::MatrixHadamarkProduct(delta_array, sub_array, delta_array)) {
        return -1;
    }
    
    //从输出层往前反向计算误差项 和梯度
    for (int i = fc_layers_array_.size() - 1; i >= 0; i--) {
        if (i == fc_layers_array_.size() - 1) {
            fc_layers_array_[i]->Backward(delta_array);
        } else {
            //用后一层的delta array 去得到本层的delta array和本层的权重梯度 偏置梯度
            fc_layers_array_[i]->Backward(fc_layers_array_[i + 1]->get_delta_array());
        }
    }
    
    return 0;
}

//利用梯度下降优化算法 更新网络的权值
void NeuralNetwork::UpdateWeights(float learning_rate) {
    for (auto fc_layer : fc_layers_array_) {
        fc_layer->UpdateWeights(learning_rate);
    }
}

//打印权重数组
void NeuralNetwork::Dump() const noexcept {
    for (auto fc_layer : fc_layers_array_) {
        fc_layer->Dump();
    }
}

//损失函数  计算均方误差 
float NeuralNetwork::Loss() const noexcept {

}

}    //namespace dnn
