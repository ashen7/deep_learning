/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年1月3日 20时32分52秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <time.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <chrono>
#include <ratio>
#include <thread>
#include <atomic>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "neural_network.h"
#include "full_connected_layer.h"
#include "utility/normalizer.hpp"
#include "utility/matrix.hpp"

//gfalgs 
DEFINE_double(learning_rate, 0.1, "AI model train learning rate");
DEFINE_int32(epoch, 0, "AI model train iterator epoch");
DEFINE_int32(mnist_training_data_szie, 10000, "Mnist Data Set Training Picture Number");
DEFINE_int32(mnist_test_data_szie, 1000, "Mnist Data Set Test Picture Number");

//global variable
const size_t kMnistImageHeight = 28;
const size_t kMnistImageWidth = 28;
const size_t kMnistImageSize = kMnistImageHeight * kMnistImageWidth;
const size_t kInputLayerNode = kMnistImageHeight * kMnistImageWidth;
const size_t kHiddenLayerNode = 300;
const size_t kOutputLayerNode = 10;
std::atomic<bool> stop_flag(false);

//训练和评估 
static int TrainAndEvaluate(std::string mnist_path);

//评估 
static double Evaluate(std::vector<std::vector<std::vector<uint8_t>>>& mnist_test_sample_data_set, 
                       std::vector<std::vector<std::vector<double>>>& mnist_test_label_data_set);

//训练
static int Train(std::string mnist_path);

//测试
static int Test(std::string mnist_path);

//得到训练集
static int GetMnistTrainingDataSet(std::string mnist_path, 
                                   size_t trainging_picture_number, 
                                   std::vector<std::vector<std::vector<uint8_t>>>& mnist_sample_data_set, 
                                   std::vector<std::vector<std::vector<double>>>& mnist_label_data_set);
            
//得到测试集
static int GetMnistTestDataSet(std::string mnist_path, 
                               size_t test_picture_number, 
                               std::vector<std::vector<std::vector<uint8_t>>>& mnist_sample_data_set, 
                               std::vector<std::vector<std::vector<double>>>& mnist_label_data_set);

//导入样本
static int LoadMnistImage(std::string mnist_image_file, 
                          size_t sample_picture_number, 
                          std::vector<std::vector<std::vector<uint8_t>>>& mnist_sample_data_set);

//导入标签
static int LoadMnistLabel(std::string mnist_label_file, 
                          size_t label_picture_number, 
                          std::vector<std::vector<std::vector<double>>>& mnist_label_data_set);

//得到一个样本
static void GetOneImageData(const std::shared_ptr<uint8_t> mnist_image_data, 
                            size_t current_image_number, 
                            std::vector<std::vector<uint8_t>>& image_data);

//梯度测试
static int TestGradient();

//得到结果
static int GetPredictResult(const std::vector<std::vector<double>>& output_array);


//得到训练集
int GetMnistTrainingDataSet(std::string mnist_path, 
                            size_t trainging_picture_number, 
                            std::vector<std::vector<std::vector<uint8_t>>>& mnist_sample_data_set, 
                            std::vector<std::vector<std::vector<double>>>& mnist_label_data_set) {
    std::string mnist_trainging_sample = mnist_path + "train-images-idx3-ubyte";
    std::string mnist_trainging_label = mnist_path + "train-labels-idx1-ubyte";
    
    if (-1 == LoadMnistImage(mnist_trainging_sample, 
                             trainging_picture_number, 
                             mnist_sample_data_set)) {
        LOG(ERROR) << "load mnist training sample failed...";
        return -1;
    }

    if (-1 == LoadMnistLabel(mnist_trainging_label, 
                             trainging_picture_number,
                             mnist_label_data_set)) {
        LOG(ERROR) << "load mnist training label failed...";
        return -1;
    }

    LOG(INFO) << "successfully load mnist training data set, load picture: " 
              << trainging_picture_number;
    return 0;
}

//得到测试集
static int GetMnistTestDataSet(std::string mnist_path, 
                               size_t test_picture_number, 
                               std::vector<std::vector<std::vector<uint8_t>>>& mnist_sample_data_set, 
                               std::vector<std::vector<std::vector<double>>>& mnist_label_data_set) {
    std::string mnist_test_sample = mnist_path + "t10k-images-idx3-ubyte";
    std::string mnist_test_label = mnist_path + "t10k-labels-idx1-ubyte";
    
    if (-1 == LoadMnistImage(mnist_test_sample, 
                             test_picture_number, 
                             mnist_sample_data_set)) {
        LOG(ERROR) << "load mnist test sample failed...";
        return -1;
    }

    if (-1 == LoadMnistLabel(mnist_test_label, 
                             test_picture_number,
                             mnist_label_data_set)) {
        LOG(ERROR) << "load mnist test label failed...";
        return -1;
    }

    LOG(INFO) << "successfully load mnist test data set, load picture: "
              << test_picture_number;
    return 0;
}

//导入样本
int LoadMnistImage(std::string mnist_image_file, 
                   size_t sample_picture_number, 
                   std::vector<std::vector<std::vector<uint8_t>>>& mnist_sample_data_set) {
    std::ifstream mnist_image;
    mnist_image.open(mnist_image_file.c_str(), std::ios::in | std::ios::binary);
    if (!mnist_image.is_open()) {
        LOG(ERROR) << "open file failed, filename is :" << mnist_image_file.c_str();
        return -1;
    }

    //shared_ptr存数组类型 要自己写删除器
    size_t mnist_image_data_size = 16 + sample_picture_number * kMnistImageSize;
    std::shared_ptr<uint8_t> mnist_image_data(new uint8_t[mnist_image_data_size], [](uint8_t* data) {
                                              delete []data; });
    //把数据读入智能指针中
    while (!mnist_image.eof()) {
        mnist_image.read(reinterpret_cast<char*>(mnist_image_data.get()), mnist_image_data_size);
    }
    mnist_image.close();
    
    if (0 != mnist_sample_data_set.size()) {
        mnist_sample_data_set.clear();
    }

    //最后都加入到这个数组中去
    mnist_sample_data_set.reserve(sample_picture_number);
    for (int i = 0; i < sample_picture_number; i++) {
        //保存每一张图像
        std::vector<std::vector<uint8_t>> image_data;
        GetOneImageData(mnist_image_data, i, image_data);
        mnist_sample_data_set.push_back(image_data);
    }

    return 0;
}

//得到一个样本 图片是28*28 这里用784*1来存储 因为要给fcnn的输入层 一个样本特征对应一个神经元的输入
void GetOneImageData(const std::shared_ptr<uint8_t> mnist_image_data, 
                     size_t current_picture_count, 
                     std::vector<std::vector<uint8_t>>& image_data) {
    if (0 == image_data.size()) {
        image_data = std::vector<std::vector<uint8_t>>(kMnistImageSize, 
                                                       std::vector<uint8_t>(1));
    }
    
    //图片数据 从索引16开始 16 + 28 * 28是第一张图片 16 + 28 * 28 + 28 * 28是第二张
    size_t start_index = 16 + current_picture_count * kMnistImageSize;
    for (int i = 0; i < kMnistImageSize; i++) {
        for (int j = 0; j < 1; j++) {
            image_data[i][j] = mnist_image_data.get()[start_index++];
        }
    }
}

//导入标签
int LoadMnistLabel(std::string mnist_label_file,  
                   size_t label_picture_number, 
                   std::vector<std::vector<std::vector<double>>>& mnist_label_data_set) {
    std::ifstream mnist_label;
    mnist_label.open(mnist_label_file.c_str(), std::ios::in | std::ios::binary);
    if (!mnist_label.is_open()) {
        LOG(ERROR) << "open file failed, filename is :" << mnist_label_file.c_str();
        return -1;
    }

    //shared_ptr存数组类型 要自己写删除器
    size_t mnist_label_data_size = 8 + label_picture_number * sizeof(char);
    std::shared_ptr<uint8_t> mnist_label_data(new uint8_t[mnist_label_data_size], [](uint8_t* data) {
                                              delete []data; });
    //把数据读入指针指针中
    while (!mnist_label.eof()) {
        mnist_label.read(reinterpret_cast<char*>(mnist_label_data.get()), mnist_label_data_size);
    }
    mnist_label.close();
    
    if (0 != mnist_label_data_set.size()) {
        mnist_label_data_set.clear();
    }

    //最后都加入到这个数组中去
    mnist_label_data_set.reserve(label_picture_number);
    for (int i = 0; i < label_picture_number; i++) {
        //从索引8开始 每个值是样本对应的label值
        uint8_t label_value = mnist_label_data.get()[8 + i];
        //保存每一张图像
        std::vector<std::vector<double>> label_data;
        size_t label_data_rows = kOutputLayerNode;
        size_t label_data_cols = 1;

        utility::Normalizer::Normalize(label_value, label_data_rows, label_data_cols, label_data);
        mnist_label_data_set.push_back(label_data);
    }

    return 0;
}

//梯度检查
int TestGradient() {
    std::string mnist_path = "./mnist/";

    //得到训练数据集
    std::vector<std::vector<std::vector<uint8_t>>> mnist_training_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_training_label_data_set;
    if (-1 == GetMnistTrainingDataSet(mnist_path, 10, 
                                      mnist_training_sample_data_set, 
                                      mnist_training_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

    //得到测试数据集
    std::vector<std::vector<std::vector<uint8_t>>> mnist_test_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_test_label_data_set;
    if (-1 == GetMnistTestDataSet(mnist_path, 10, 
                                  mnist_test_sample_data_set, 
                                  mnist_test_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

    //梯度检查
    SingletonNeuralNetwork::Instance().GradientCheck(mnist_training_sample_data_set[8], 
                                                     mnist_training_label_data_set[8]);

    return 0;
}

//训练和评估  学习策略
int TrainAndEvaluate(std::string mnist_path) {
    //得到训练数据集
    std::vector<std::vector<std::vector<uint8_t>>> mnist_training_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_training_label_data_set;
    if (-1 == GetMnistTrainingDataSet(mnist_path, FLAGS_mnist_training_data_szie, 
                                      mnist_training_sample_data_set, 
                                      mnist_training_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

    //得到测试数据集
    std::vector<std::vector<std::vector<uint8_t>>> mnist_test_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_test_label_data_set;
    if (-1 == GetMnistTestDataSet(mnist_path, FLAGS_mnist_test_data_szie, 
                                  mnist_test_sample_data_set, 
                                  mnist_test_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

    //记录上次的测试错误率 如果本次测试错误率高于上次 证明模型过拟合了 退出训练   
    double last_error_ratio = 1.0;

    LOG(INFO) << "==============开始训练===============";
    while (true) {
        FLAGS_epoch++;
        //每次训练完成1轮后 打印一下当前情况
        SingletonNeuralNetwork::Instance().Train(mnist_training_sample_data_set, 
                                                 mnist_training_label_data_set, 
                                                 1, 
                                                 FLAGS_learning_rate);
        //得到当前时间
        char now[60] = { 0 };
        calculate::time::GetCurrentTime(now);
        //拿一个样本和标签 算loss
        std::vector<std::vector<double>> output_array;
        SingletonNeuralNetwork::Instance().Predict(mnist_test_sample_data_set[8], 
                                                   output_array);
        double loss = SingletonNeuralNetwork::Instance().Loss(output_array, 
                                                              mnist_test_label_data_set[8]);
        LOG(INFO) << now << " epoch " << FLAGS_epoch
                  << " finished, loss: " << loss;
        
        //每训练10轮 测试一次
        if (0 == FLAGS_epoch % 5) {
            double current_error_ratio = Evaluate(mnist_test_sample_data_set, 
                                                  mnist_test_label_data_set);
            char _now[60] = { 0 };
            calculate::time::GetCurrentTime(_now);
            LOG(WARNING) << _now << " after epoch " << FLAGS_epoch
                         << ", error ratio is: %" << current_error_ratio * 100;
            if (current_error_ratio > last_error_ratio) {
                LOG(INFO) << "==============训练结束===============";
                break;
            } else {
                last_error_ratio = current_error_ratio;
            }
        }
    }

    return 0;
}

//评估 
double Evaluate(std::vector<std::vector<std::vector<uint8_t>>>& mnist_test_sample_data_set, 
                std::vector<std::vector<std::vector<double>>>& mnist_test_label_data_set) {
    double error = 0.0;
    size_t test_data_set_total = mnist_test_sample_data_set.size();
    int label = -1;
    int output = -1;
    std::vector<std::vector<double>> output_array;
    //遍历测试数据集 做预测 查看结果 
    for (int i = 0; i < test_data_set_total; i++) {
        label = GetPredictResult(mnist_test_label_data_set[i]);
        SingletonNeuralNetwork::Instance().Predict(mnist_test_sample_data_set[i], 
                                                   output_array);
        output = GetPredictResult(output_array);
        //LOG(INFO) << "预测值: " << output << ", 标签: " << label;
        if (label != output) {
            error += 1.0;
        }
    }
    
    return error / test_data_set_total;
}

//训练模型 梯度下降优化算法 更新权值
int Train(std::string mnist_path) {
    //得到训练数据集
    std::vector<std::vector<std::vector<uint8_t>>> mnist_training_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_training_label_data_set;
    if (-1 == GetMnistTrainingDataSet(mnist_path, FLAGS_mnist_training_data_szie, 
                                      mnist_training_sample_data_set, 
                                      mnist_training_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }
    
    int epoch = 5;
    LOG(INFO) << "==============开始训练===============";
    while (true) {
        FLAGS_epoch++;
        //每次训练完成1轮后 打印一下当前情况
        SingletonNeuralNetwork::Instance().Train(mnist_training_sample_data_set, 
                                                 mnist_training_label_data_set, 
                                                 1, 
                                                 FLAGS_learning_rate);
        //得到当前时间
        char now[60] = { 0 };
        calculate::time::GetCurrentTime(now);
        //拿一个样本和标签 算loss
        std::vector<std::vector<double>> output_array;
        SingletonNeuralNetwork::Instance().Predict(mnist_training_sample_data_set[8], 
                                                   output_array);
        double loss = SingletonNeuralNetwork::Instance().Loss(output_array, 
                                                              mnist_training_label_data_set[8]);
        LOG(INFO) << now << " epoch " << FLAGS_epoch
                  << " finished, loss: " << loss;

        if (FLAGS_epoch == epoch) {
            break;
        }
    }

    LOG(INFO) << "==============训练结束===============";
    
    return 0;
}

//测试模型   用错误率来对网络进行评估
int Test(std::string mnist_path) {
    //得到测试数据集
    std::vector<std::vector<std::vector<uint8_t>>> mnist_test_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_test_label_data_set;
    if (-1 == GetMnistTestDataSet(mnist_path, FLAGS_mnist_test_data_szie, 
                                  mnist_test_sample_data_set, 
                                  mnist_test_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

    double error = 0.0;
    size_t test_data_set_total = mnist_test_sample_data_set.size();
    int label = -1;
    int output = -1;
    std::vector<std::vector<double>> output_array;
    //遍历测试数据集 做预测 查看结果 
    for (int i = 0; i < test_data_set_total; i++) {
        label = GetPredictResult(mnist_test_label_data_set[i]);
        SingletonNeuralNetwork::Instance().Predict(mnist_test_sample_data_set[i], 
                                                   output_array);
        output = GetPredictResult(output_array);
        if (label != output) {
            error += 1.0;
        }
    }
    
    double error_ratio = error / test_data_set_total;
    char now[60] = { 0 };
    calculate::time::GetCurrentTime(now);
    LOG(WARNING) << now << " after epoch " << FLAGS_epoch
                 << ", error ratio is: %" << error_ratio * 100;

    return 0;
}


//10行1列 每个值就是一个类别的预测值 取最大的值就是预测结果
int GetPredictResult(const std::vector<std::vector<double>>& output_array) {
    double max_value = 0.0; 
    int max_value_index = 0;
    for (int i = 0; i < output_array.size(); i++) {
        for (int j = 0; j < output_array[i].size(); j++) {
            if (output_array[i][j] > max_value) {
                max_value = output_array[i][j];
                max_value_index = i;
            }
        }
    }
    
    return max_value_index;
}


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;

    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

    //初始化fcnn  三层节点 784 300 10
    std::vector<size_t> fc_layer_nodes_array{kInputLayerNode, kHiddenLayerNode, kOutputLayerNode};
    SingletonNeuralNetwork::Instance().Initialize(fc_layer_nodes_array);

    std::string mnist_path = "./mnist/";
    TrainAndEvaluate(mnist_path);
    //TestGradient();
    //Train(mnist_path);
    //Test(mnist_path);
    
    //记录结束时间
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //设置单位为秒
    std::chrono::duration<int, std::ratio<1, 1>> sec = std::chrono::duration_cast<
                                                       std::chrono::seconds>(end - begin);
    //打印耗时
    LOG(INFO) << "programming is exiting, the total time is: " << sec.count() << "s";
    google::ShutdownGoogleLogging();
    
    return 0;

}

