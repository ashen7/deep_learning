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
#include <time.h>
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
#include <omp.h>

#include "convolutional_layer.h"
#include "max_pooling_layer.h"
#include "utility/normalizer.hpp"
#include "utility/matrix_math_function.hpp"

//gfalgs 
DEFINE_double(learning_rate, 0.001, "AI model train learning rate");

//测试卷积层
static int TestConvolutionLayer();
static int TestConvolutionLayerBP();
static int TestGradient();
static void InitializeConvolutionLayer(std::vector<std::vector<std::vector<double>>>& input_array, 
                                       std::vector<std::vector<std::vector<double>>>& sensitivity_array);

//测试最大池化层
static int TestMaxPoolingLayer();
static int TestMaxPoolingLayerBP();
static void InitializeMaxPoolingLayer(std::vector<std::vector<std::vector<double>>>& input_array, 
                                      std::vector<std::vector<std::vector<double>>>& sensitivity_array);


//测试CNN前向计算
int TestConvolutionLayer() {
    std::vector<std::vector<std::vector<double>>> input_array;
    std::vector<std::vector<std::vector<double>>> sensitivity_array;
    //前向传播
    InitializeConvolutionLayer(input_array, sensitivity_array);
    SingletonConvLayer::Instance().Dump();
    if (-1 == SingletonConvLayer::Instance().Forward(input_array)) {
        LOG(ERROR) << "conv layer forward failed";
        return -1;
    }
    
    LOG(INFO) << "input array:";
    Matrix::MatrixShow(input_array);
    LOG(INFO) << "output feature map:";
    Matrix::MatrixShow(SingletonConvLayer::Instance().get_output_array());

    return 0;
}

//测试CNN反向传播
int TestConvolutionLayerBP() {
    std::vector<std::vector<std::vector<double>>> input_array;
    std::vector<std::vector<std::vector<double>>> sensitivity_array;
    //反向传播
    InitializeConvolutionLayer(input_array, sensitivity_array);
    if (-1 == SingletonConvLayer::Instance().Backward(input_array,  
                                                      sensitivity_array)) {
        LOG(ERROR) << "conv layer backward failed";
        return -1;
    }
    //更新权重
    SingletonConvLayer::Instance().UpdateWeights();
    SingletonConvLayer::Instance().Dump();

    LOG(INFO) << "input array:";
    Matrix::MatrixShow(input_array);
    LOG(INFO) << "sensitivity map:";
    Matrix::MatrixShow(sensitivity_array);
    LOG(INFO) << "delta array:";
    Matrix::MatrixShow(SingletonConvLayer::Instance().get_delta_array());
    return 0;
}

//梯度检查
int TestGradient() {
    std::vector<std::vector<std::vector<double>>> input_array;
    std::vector<std::vector<std::vector<double>>> sensitivity_array;
    //梯度检查
    InitializeConvolutionLayer(input_array, sensitivity_array);
    if (-1 == SingletonConvLayer::Instance().GradientCheck(input_array)) {
        LOG(ERROR) << "conv layer gradient check failed";
        return -1;
    }
    
    return 0;
}

//初始化卷积层
void InitializeConvolutionLayer(std::vector<std::vector<std::vector<double>>>& input_array, 
                                std::vector<std::vector<std::vector<double>>>& sensitivity_array) {
    //输入图像 3 * 5 * 5
    input_array = std::vector<std::vector<std::vector<double>>>{
        {{0, 1, 1, 0, 2}, 
         {2, 2, 2, 2, 1}, 
         {1, 0, 0, 2, 0}, 
         {0, 1, 1, 0, 0}, 
         {1, 2, 0, 0, 2}}, 

        {{1, 0, 2, 2, 0},
         {0, 0, 0, 2, 0}, 
         {1, 2, 1, 2, 1}, 
         {1, 0, 0, 0, 0}, 
         {1, 2, 1, 1, 1}}, 

        {{2, 1, 2, 0, 0}, 
         {1, 0, 0, 1, 0}, 
         {0, 2, 1, 0, 1}, 
         {0, 1, 2, 2, 2}, 
         {2, 1, 0, 0, 1}}
    };

    //构造卷积层 输入3 * 5 * 5 filter2 * 3 * 3 补0填充1 filter移动步长2
    size_t input_height = 5;
    size_t input_width = 5;
    size_t channel_number = 3;
    size_t filter_height = 3;
    size_t filter_width = 3;
    size_t filter_number = 2;
    size_t zero_padding = 1;
    size_t stride = 2;
    //double learning_rate = 0.001;
    SingletonConvLayer::Instance().Initialize(input_height, input_width, channel_number, 
                                              filter_height, filter_width, filter_number, 
                                              zero_padding, stride, FLAGS_learning_rate);
    
    //给每个filter的权重 和 偏置 重新设置值
    std::vector<std::vector<std::vector<std::vector<double>>>> filters_weights;
    std::vector<double> filters_biases;
    filters_weights.reserve(filter_number);
    filters_biases.reserve(filter_number);
    std::vector<std::vector<std::vector<double>>> filter_weights_1{
        {{-1, 1, 0 }, 
         { 0, 1, 0 }, 
         { 0, 1, 1 }}, 

        {{-1, -1, 0},
         { 0,  0, 0}, 
         { 0, -1, 0}}, 

        {{0,  0, -1}, 
         {0,  1,  0}, 
         {1, -1, -1}}
    };
    std::vector<std::vector<std::vector<double>>> filter_weights_2{
        {{ 1,  1,-1}, 
         {-1, -1, 1}, 
         { 0, -1, 1}}, 

        {{0,   1, 0},
         {-1,  0,-1}, 
         {-1,  1, 0}}, 

        {{-1,  0, 0}, 
         {-1,  0, 1}, 
         {-1,  0, 0}}
    };
    filters_weights.push_back(filter_weights_1);
    filters_weights.push_back(filter_weights_2);
    double filter_bias_1 = 1;
    filters_biases.push_back(filter_bias_1);

    SingletonConvLayer::Instance().set_filters(filters_weights, filters_biases);

    //sensitivity map 敏感图(误差项) shape和特征图shape一致
    sensitivity_array = std::vector<std::vector<std::vector<double>>>{
        {{ 0, 1, 1 }, 
         { 2, 2, 2 }, 
         { 1, 0, 0 }}, 

        {{ 1, 0, 2 },
         { 0, 0, 0 }, 
         { 1, 2, 1 }}, 
    };
}

//测试最大池化层前向传播
int TestMaxPoolingLayer() {
    std::vector<std::vector<std::vector<double>>> input_array;
    std::vector<std::vector<std::vector<double>>> sensitivity_array;
    InitializeMaxPoolingLayer(input_array, sensitivity_array);
    if (-1 == SingletonPoolLayer::Instance().Forward(input_array)) {
        LOG(ERROR) << "pool layer forward failed";
        return -1;
    }
    
    LOG(INFO) << "input array:";
    Matrix::MatrixShow(input_array);
    LOG(INFO) << "output array:";
    Matrix::MatrixShow(SingletonPoolLayer::Instance().get_output_array());
    return 0;
}

//测试最大池化层反向传播
int TestMaxPoolingLayerBP() {
    std::vector<std::vector<std::vector<double>>> input_array;
    std::vector<std::vector<std::vector<double>>> sensitivity_array;
    InitializeMaxPoolingLayer(input_array, sensitivity_array);
    if (-1 == SingletonPoolLayer::Instance().Forward(input_array)) {
        LOG(ERROR) << "pool layer forward failed";
        return -1;
    }
    if (-1 == SingletonPoolLayer::Instance().Backward(input_array, sensitivity_array)) {
        LOG(ERROR) << "pool layer backward failed";
        return -1;
    }

    LOG(INFO) << "input array:";
    Matrix::MatrixShow(input_array);
    LOG(INFO) << "sensitivity map:";
    Matrix::MatrixShow(sensitivity_array);
    LOG(INFO) << "delta array(upstream sensitivity map):";
    Matrix::MatrixShow(SingletonPoolLayer::Instance().get_delta_array());
    return 0;
}

//初始化池化层
void InitializeMaxPoolingLayer(std::vector<std::vector<std::vector<double>>>& input_array, 
                               std::vector<std::vector<std::vector<double>>>& sensitivity_array) {
    //输入2 * 4 * 4
    input_array = std::vector<std::vector<std::vector<double>>>{
        {{ 1, 1, 2, 4 }, 
         { 5, 6, 7, 8 }, 
         { 3, 2, 1, 0 }, 
         { 1, 2, 3, 4 }}, 

        {{ 0, 1, 2, 3 },
         { 4, 5, 6, 7 }, 
         { 8, 9, 0, 1 }, 
         { 3, 4, 5, 6 }}, 
    };

    //构造池化层
    int input_height = 4;
    int input_width = 4;
    int channel_number = 2;
    int filter_height = 2;
    int filter_width = 2;
    int stride = 2;
    
    sensitivity_array = std::vector<std::vector<std::vector<double>>>{
        {{ 1, 2 }, 
         { 2, 4 }}, 

        {{ 3, 5 },
         { 8, 2 }}, 
    };
    
    //初始化
    SingletonPoolLayer::Instance().Initialize(input_height, input_width, 
                                              channel_number, filter_height, 
                                              filter_width, stride);
}

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;

    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

    TestConvolutionLayer();
    //TestConvolutionLayerBP();
    //TestGradient();
    //TestMaxPoolingLayer();
    //TestMaxPoolingLayerBP();

    //记录结束时间
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //设置单位为毫秒
    std::chrono::duration<int, std::milli> milli = std::chrono::duration_cast<
                                                   std::chrono::milliseconds>(end - begin);
    //打印耗时
    LOG(INFO) << "programming is exiting, the total time is: " << milli.count() << "ms";
    google::ShutdownGoogleLogging();
    
    return 0;

}

