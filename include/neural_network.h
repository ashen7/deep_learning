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

//#include "full_connected_layer.h"

namespace dnn {

//这里使用类的前置声明就可以了
class FullConnectedLayer;

//神经网络类
class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator =(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) = default;
    NeuralNetwork& operator =(NeuralNetwork&&) = default;

public:
    void Initialize(const std::vector<size_t>& fc_layer_nodes_array);
    int Train(const std::vector<std::vector<std::vector<float>>>& training_data_set, 
              const std::vector<std::vector<std::vector<float>>>& labels,
              int epoch, float learning_rate);
    int Predict(const std::vector<std::vector<float>>& input_array, 
                std::vector<std::vector<float>>& output_array);
    int CalcGradient(const std::vector<std::vector<float>>& output_array, 
                     const std::vector<std::vector<float>>& label);
    void UpdateWeights(float learning_rate);
    void Dump() const noexcept;
    float Loss() const noexcept;

protected:
    //内部函数
    int TrainOneSample(const std::vector<std::vector<float>>& input_array, 
                       const std::vector<std::vector<float>>& label, 
                       float learning_rate);

private:
    std::vector<std::shared_ptr<FullConnectedLayer>> fc_layers_array_;
};


}      //namespace dnn

#endif //DNN_NEURAL_NETWORK_H_

