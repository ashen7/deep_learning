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
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <chrono>
#include <thread>
#include <ratio>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "neural_network.h"
#include "full_connected_layer.h"
#include "utility/normalizer.hpp"
#include "utility/matrix.hpp"

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
    
    //三层节点 8 10 8
    std::vector<size_t> fc_layer_nodes_array{8, 10, 8};
    SingletonNeuralNetwork::Instance().Initialize(fc_layer_nodes_array);
    
    float learning_rate = 0.3;   //学习率
    size_t batch_size = 10;      //batch大小
    size_t epoch = 1;           //迭代轮数
    
    std::vector<std::vector<float>> output_array;
    for (int i = 0; i < epoch; i++) {
        //训练完成1轮
        SingletonNeuralNetwork::Instance().Train(data_set, labels, batch_size, learning_rate);
        //打印loss 均方误差的值 看看是否收敛
        SingletonNeuralNetwork::Instance().Predict(data_set[data_set.size() - 1], output_array);
        LOG(WARNING) << "after epoch " << i << " loss: " 
                     << SingletonNeuralNetwork::Instance().Loss(output_array, labels[labels.size() - 1]);
        learning_rate /= 2.0;
    }

    //neural_network.Predict(data_set[128], output_array);
    //number = utility::Normalizer::Denormalize(output_array);
    //calculate::matrix::MatrixShow(output_array);
    //LOG(WARNING) << "预测的值为: " << (int)number;

    SingletonNeuralNetwork::Instance().Dump();
}


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;

    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();
    //TestFCNN();
    std::vector<std::vector<float>> a;
    std::vector<std::vector<float>> b;

    TestFCNN();

    //记录结束时间
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //设置单位为秒
    std::chrono::duration<int, std::ratio<1, 1>> sec = std::chrono::duration_cast<
                                                       std::chrono::seconds>(end - begin);
    //设置单位为毫秒
    std::chrono::duration<int, std::milli> milli = std::chrono::duration_cast<
                                                   std::chrono::milliseconds>(end - begin);
    //打印耗时
    //LOG(WARNING) << "程序退出, 总共耗时: " << sec.count() << "s";
    LOG(WARNING) << "程序退出, 总共耗时: " << milli.count() << "ms";
    google::ShutdownGoogleLogging();
    
    return 0;

}

