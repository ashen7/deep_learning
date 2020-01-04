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

#include "utility/singleton.hpp"

namespace dnn {

//这里使用类的前置声明就可以了
class FullConnectedLayer;

//神经网络类
class NeuralNetwork {
public:
    //这里只用到float类型 设置一个类型别名
    typedef std::vector<std::vector<float>> Matrix2d;
    typedef std::vector<std::vector<std::vector<float>>> Matrix3d;

    NeuralNetwork();
    ~NeuralNetwork();
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator =(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) = default;
    NeuralNetwork& operator =(NeuralNetwork&&) = default;

public:
    void Initialize(const std::vector<size_t>& fc_layer_nodes_array);
    int Train(const Matrix3d& training_data_set, 
              const Matrix3d& labels,
              int epoch, float learning_rate);
    int Predict(const Matrix2d& input_array, 
                Matrix2d& output_array);
    int CalcGradient(const Matrix2d& output_array, 
                     const Matrix2d& label);
    void UpdateWeights(float learning_rate);
    void Dump() const noexcept;
    float Loss(const Matrix2d& output_array, 
               const Matrix2d& label) const noexcept;
    int GradientCheck(const Matrix2d& sample, 
                      const Matrix2d& label);

protected:
    //内部函数
    int TrainOneSample(const Matrix2d& sample, 
                       const Matrix2d& label, 
                       float learning_rate);

private:
    std::vector<std::shared_ptr<FullConnectedLayer>> fc_layers_array_;
};

}      //namespace dnn

//单例模式
typedef typename utility::Singleton<dnn::NeuralNetwork> SingletonNeuralNetwork;




#endif //DNN_NEURAL_NETWORK_H_

