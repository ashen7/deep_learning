/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年12月29日 20时32分52秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>
#include <random>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "utility/neural_network.h"
#include "utility/full_connected_layer.h"
#include "utility/matrix_math_function.hpp"
#include "utility/normalizer.hpp"

//构造训练数据集
static void TrainDataSet(std::vector<std::vector<std::vector<float>>>& data_set, 
                         std::vector<std::vector<std::vector<float>>>& labels) {
    data_set.reserve(256);
    labels.reserve(256);

    for (int i = 0; i < 256; i++) {
        std::vector<std::vector<float>> data(8, std::vector<float>(1));
        //归一化
        utility::Normalizer::Normalize(i, data);
        data_set.push_back(data);
        labels.push_back(data);
    }
}

//测试FCNN
static void TestFCNN() {
    //存储 输入特征x (8行1列)的列向量  
    std::vector<std::vector<std::vector<float>>> data_set;
    //存储 真实值y   (8行1列)的列向量
    std::vector<std::vector<std::vector<float>>> labels;
    TrainDataSet(data_set, labels);
    
    //神经网络对象
    dnn::NeuralNetwork neural_network;
    //三层节点 8 10 8
    std::vector<size_t> fc_layer_nodes_array{8, 2, 8};
    neural_network.Initialize(fc_layer_nodes_array);
    
    float learning_rate = 0.3;  //学习率
    size_t batch_size = 20;      //batch大小
    size_t epoch = 10;           //迭代轮数
    
    std::vector<std::vector<float>> output_array;
    
    for (int i = 0; i < epoch; i++) {
        //训练1轮
        //neural_network.train(data_set, labels, batch_size, learning_rate);
    }
}


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;

    //TestFCNN();
    std::vector<std::vector<float>> a;
    std::vector<std::vector<float>> b;
    calculate::random::Uniform(0, 6, 3, 3, a);
    calculate::matrix::MatrixShow(a);
    calculate::matrix::TransposeMatrix(a, b);
    calculate::matrix::MatrixShow(b);
    
    google::ShutdownGoogleLogging();
    
    return 0;
}
