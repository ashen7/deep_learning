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
#include "utility/normalizer.hpp"
#include "utility/matrix_math_function.hpp"

//gfalgs 
DEFINE_double(learning_rate, 0.3, "AI model train learning rate");
DEFINE_int32(batch_size, 20, "AI model train batch size");
DEFINE_int32(epoch, 10, "AI model train iterator epoch");

static void TestConvolution();
static void InitializeConvolution();
static void TestGradient();

//测试FCNN
void TestConvolution() {
    InitializeConvolution();
}

void InitializeConvolution() {
    //输入图像 3 * 5 * 5
    std::vector<std::vector<std::vector<double>>> input_array{
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

    //构造卷积层
    size_t input_height = 5;
    size_t input_width = 5;
    size_t channel_number = 3;
    size_t filter_height = 3;
    size_t filter_width = 3;
    size_t filter_number = 2;
    size_t zero_padding = 1;
    size_t stride = 2;
    double learning_rate = 0.001;
    SingletonConvLayer::Instance().Initialize(input_height, input_width, channel_number, 
                                              filter_height, filter_width, filter_number, 
                                              zero_padding, stride, learning_rate);
    
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
    SingletonConvLayer::Instance().Dump();
    std::vector<std::vector<std::vector<double>>> cc;
    Matrix::ZeroPadding(filter_weights_1, 2, cc);
    Matrix::MatrixShow(cc);
}

//梯度检查
void TestGradient() {
}

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;

    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

    TestConvolution();
    //TestGradient();

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

