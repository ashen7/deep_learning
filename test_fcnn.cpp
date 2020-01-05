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

//gfalgs 
DEFINE_double(learning_rate, 0.3, "AI model train learning rate");
DEFINE_int32(batch_size, 20, "AI model train batch size");
DEFINE_int32(epoch, 10, "AI model train iterator epoch");

static void TrainDataSet(std::vector<std::vector<std::vector<double>>>& data_set, 
                         std::vector<std::vector<std::vector<double>>>& labels);
static void TestFCNN();
static void TestGradient();
static double GetCorrectRatio();

//构造训练数据集
void TrainDataSet(std::vector<std::vector<std::vector<double>>>& data_set, 
                  std::vector<std::vector<std::vector<double>>>& labels) {
    data_set.reserve(256);
    labels.reserve(256);

    for (int i = 0; i < 256; i++) {
        std::vector<std::vector<double>> data(8, std::vector<double>(1));
        //归一化
        utility::Normalizer::Normalize(i, data);
        data_set.push_back(data);
        labels.push_back(data);
    }
}

//测试FCNN
void TestFCNN() {
    //存储 输入特征x (8行1列)的列向量  
    std::vector<std::vector<std::vector<double>>> data_set;
    //存储 真实值y   (8行1列)的列向量
    std::vector<std::vector<std::vector<double>>> labels;
    TrainDataSet(data_set, labels);
    
    //三层节点 8 10 8
    std::vector<size_t> fc_layer_nodes_array{8, 10, 8};
    SingletonNeuralNetwork::Instance().Initialize(fc_layer_nodes_array);
    
    std::vector<std::vector<double>> output_array;
    for (int i = 0; i < FLAGS_epoch; i++) {
        //训练完成1轮
        SingletonNeuralNetwork::Instance().Train(data_set, labels, FLAGS_batch_size, FLAGS_learning_rate);
        
        //打印loss 均方误差的值 看看是否收敛
        SingletonNeuralNetwork::Instance().Predict(data_set[100], output_array);
        double correct_ratio = GetCorrectRatio();
        LOG(WARNING) << "after epoch " << i << " loss: " 
                     << SingletonNeuralNetwork::Instance().Loss(output_array, labels[100])
                     << ", correct ratio: %" << correct_ratio;
        //得到当前的正确率
        FLAGS_learning_rate /= 2.0;
    }

    //SingletonNeuralNetwork::Instance().Dump();
}

//梯度检查
void TestGradient() {
    //存储 输入特征x (8行1列)的列向量  
    std::vector<std::vector<std::vector<double>>> data_set;
    //存储 真实值y   (8行1列)的列向量
    std::vector<std::vector<std::vector<double>>> labels;
    TrainDataSet(data_set, labels);
    
    //初始化神经网络
    std::vector<size_t> fc_layer_nodes_array{8, 10, 8};
    SingletonNeuralNetwork::Instance().Initialize(fc_layer_nodes_array);

    //梯度检查
    SingletonNeuralNetwork::Instance().GradientCheck(data_set[2], labels[2]);
}

//用正确率来评估网络
double GetCorrectRatio() {
    double correct = 0.0;
    std::vector<std::vector<double>> input_array;
    std::vector<std::vector<double>> output_array;

    for (int i = 0; i < 256; i++) {
        utility::Normalizer::Normalize(i, input_array);
        SingletonNeuralNetwork::Instance().Predict(input_array, output_array);
        if (i == int(utility::Normalizer::Denormalize(output_array))) {
            correct += 1.0;
        }
    }
    
    return correct / 256 * 100;
}

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;

    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

    TestFCNN();
    //TestGradient();

    //记录结束时间
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //设置单位为秒
    std::chrono::duration<int, std::ratio<1, 1>> sec = std::chrono::duration_cast<
                                                       std::chrono::seconds>(end - begin);
    //设置单位为毫秒
    std::chrono::duration<int, std::milli> milli = std::chrono::duration_cast<
                                                   std::chrono::milliseconds>(end - begin);
    //打印耗时
    LOG(WARNING) << "程序退出, 总共耗时: " << milli.count() << "ms";
    google::ShutdownGoogleLogging();
    
    return 0;

}

