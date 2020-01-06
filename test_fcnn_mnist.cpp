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

//global variable
const size_t kMnistImageHeight = 28;
const size_t kMnistImageWidth = 28;
const size_t kMnistImageSize = kMnistImageHeight * kMnistImageWidth;
const size_t kInputLayerNode = kMnistImageHeight * kMnistImageWidth;
const size_t kHiddenLayerNode = 300;
const size_t kOutputLayerNode = 10;

static void GetMnistTrainingDataSet();

static void GetMnistTestDataSet();

static int LoadMnistImage(std::string mnist_filename, 
                          size_t picture_number, 
                          std::vector<std::vector<std::vector<uint8_t>>>& mnist_data_set);

static void GetOneImageData(const std::shared_ptr<uint8_t> mnist_image_data, 
                            size_t current_image_number, 
                            std::vector<std::vector<uint8_t>>& image_data);

static void TrainDataSet(std::vector<std::vector<std::vector<double>>>& data_set, 
                         std::vector<std::vector<std::vector<double>>>& labels);
static void TestFCNN();

void GetMnistTrainingDataSet() {
    
}

int LoadMnistImage(std::string mnist_filename, 
                   size_t picture_number, 
                   std::vector<std::vector<std::vector<uint8_t>>>& mnist_data_set) {
    std::ifstream mnist;
    mnist.open(mnist_filename.c_str(), std::ios::in | std::ios::binary);
    if (!mnist.is_open()) {
        LOG(ERROR) << "open file failed, filename is :" << mnist_filename.c_str();
        return -1;
    }

    //shared_ptr存数组类型 要自己写删除器
    size_t mnist_data_size = 16 + picture_number * kMnistImageSize;
    std::shared_ptr<uint8_t> mnist_image_data(new uint8_t[mnist_data_size], [](uint8_t* data) {
                                              delete []data; });
    //把数据读入指针指针中
    while (!mnist.eof()) {
        mnist.read(reinterpret_cast<char*>(mnist_image_data.get()), mnist_data_size);
    }
    
    if (0 != mnist_data_set.size()) {
        mnist_data_set.clear();
    }

    //最后都加入到这个数组中去
    mnist_data_set.reserve(picture_number);
    for (int i = 0; i < picture_number; i++) {
        //保存每一张图像
        std::vector<std::vector<uint8_t>> image_data;
        GetOneImageData(mnist_image_data, i, image_data);
        mnist_data_set.push_back(image_data);
    }

    return 0;
}

void GetOneImageData(const std::shared_ptr<uint8_t> mnist_image_data, 
                     size_t current_image_number, 
                     std::vector<std::vector<uint8_t>>& image_data) {
    if (0 == image_data.size()) {
        image_data = std::vector<std::vector<uint8_t>>(kMnistImageHeight, 
                                                       std::vector<uint8_t>(kMnistImageWidth));
    }
    
    //图片数据 从索引16开始 16 + 28 * 28是第一张图片 16 + 28 * 28 + 28 * 28是第二张
    size_t start_index = 16 + current_image_number * kMnistImageSize;
    for (int i = 0; i < kMnistImageHeight; i++) {
        for (int j = 0; j < kMnistImageWidth; j++) {
            image_data[i][j] = mnist_image_data.get()[start_index++];
        }
    }
}

int LoadMnistLabel(std::string mnist_filename, 
                   size_t picture_number, 
                   std::vector<std::vector<std::vector<uint8_t>>>& mnist_data_set) {
    std::ifstream mnist;
    mnist.open(mnist_filename.c_str(), std::ios::in | std::ios::binary);
    if (!mnist.is_open()) {
        LOG(ERROR) << "open file failed, filename is :" << mnist_filename.c_str();
        return -1;
    }

    //shared_ptr存数组类型 要自己写删除器
    size_t mnist_data_size = 16 + picture_number * kMnistImageSize;
    std::shared_ptr<uint8_t> mnist_image_data(new uint8_t[mnist_data_size], [](uint8_t* data) {
                                              delete []data; });
    //把数据读入指针指针中
    while (!mnist.eof()) {
        mnist.read(reinterpret_cast<char*>(mnist_image_data.get()), mnist_data_size);
    }
    
    if (0 != mnist_data_set.size()) {
        mnist_data_set.clear();
    }

    //最后都加入到这个数组中去
    mnist_data_set.reserve(picture_number);
    for (int i = 0; i < picture_number; i++) {
        //保存每一张图像
        std::vector<std::vector<uint8_t>> image_data;
        GetOneImageData(mnist_image_data, i, image_data);
        mnist_data_set.push_back(image_data);
    }

    return 0;
}





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
        LOG(WARNING) << "after epoch " << i << " loss: " 
                     << SingletonNeuralNetwork::Instance().Loss(output_array, labels[100]);
    }

    //SingletonNeuralNetwork::Instance().Dump();
}

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;

    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();
    
    std::string mnist_path = "./mnist/";
    std::string mnist_trainging_sample = mnist_path + "train-images-idx3-ubyte";
    std::string mnist_trainging_label = mnist_path + "train-labels-idx1-ubyte";
    std::string mnist_test_sample = mnist_path + "t10k-images-idx3-ubyte";
    std::string mnist_test_label = mnist_path + "t10k-labels-idx1-ubyte";

    std::vector<std::vector<std::vector<uint8_t>>> mnist_data_set;
    LoadMnistImage(mnist_trainging_sample, 2, mnist_data_set);
    for (auto aa : mnist_data_set) {
        ImageMatrix::ImageMatrixShow(aa);
    }
    
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

