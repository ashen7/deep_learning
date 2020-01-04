/*
 * =====================================================================================
 *
 *       Filename:  matrix_math_functions.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年12月29日 20时13分08秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef MATRIX_MATH_FUNCTIONS_HPP_
#define MATRIX_MATH_FUNCTIONS_HPP_

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>
#include <map>
#include <tuple>
#include <memory>

#include <glog/logging.h>

//模板函数 
//模板函数有两层命名空间  
namespace calculate {
namespace matrix {

// 函数声明
// 矩阵相乘  dot product 点积
template <class DataType=float>
int MatrixMultiply(const std::vector<std::vector<DataType>>& left_matrix, 
                   const std::vector<std::vector<DataType>>& right_matrix, 
                   std::vector<std::vector<DataType>>& result_matrix); 

// 矩阵对应位置相乘 hadamark积 
template <typename DataType=float>
int MatrixHadamarkProduct(const std::vector<std::vector<DataType>>& left_matrix, 
                          const std::vector<std::vector<DataType>>& right_matrix, 
                          std::vector<std::vector<DataType>>& result_matrix);

// 矩阵相加 
template <typename DataType=float>
int MatrixAdd(const std::vector<std::vector<DataType>>& add_matrix, 
              const std::vector<std::vector<DataType>>& beadd_matrix, 
              std::vector<std::vector<DataType>>& result_matrix);

// 矩阵相减
template <typename DataType=float>
int MatrixSubtract(const std::vector<std::vector<DataType>>& sub_matrix, 
                   const std::vector<std::vector<DataType>>& besub_matrix, 
                   std::vector<std::vector<DataType>>& result_matrix);

// 矩阵reshape 成几行几列  这是二维矩阵版
template <typename DataType=float>
int MatrixReshape(const std::vector<std::vector<DataType>>& source_matrix,
                  size_t rows, size_t cols, 
                  std::vector<std::vector<DataType>>& result_matrix);

// 矩阵reshape 一维矩阵转二维矩阵版
template <typename DataType=float>
int MatrixReshape(const std::vector<DataType>& source_matrix,
                  size_t rows, size_t cols, 
                  std::vector<std::vector<DataType>>& result_matrix); 

// 打印二维矩阵
template <typename DataType=float>
void MatrixShow(const std::vector<std::vector<DataType>>& matrix);

// 创建2维矩阵 初始值为0
template <typename DataType=float>
void CreateZerosMatrix(int rows, int cols, 
                       std::vector<std::vector<DataType>>& matrix);

// 创建2维矩阵 初始值为1
template <typename DataType=float>
void CreateOnesMatrix(int rows, int cols, 
                      std::vector<std::vector<DataType>>& matrix);

// 2维矩阵的装置矩阵
template <typename DataType=float>
int TransposeMatrix(const std::vector<std::vector<DataType>>& source_matrix, 
                    std::vector<std::vector<DataType>>& result_matrix);

// 1个值 乘以 一个2d矩阵
template <typename DataType=float>
void MatrixMulValue(const std::vector<std::vector<DataType>>& source_matrix, 
                    DataType value,  
                    std::vector<std::vector<DataType>>& result_matrix);

// 1个值 乘以 一个2d矩阵
template <typename DataType=float>
void ValueSubMatrix(DataType value,  
                    const std::vector<std::vector<DataType>>& source_matrix, 
                    std::vector<std::vector<DataType>>& result_matrix);

//计算2d矩阵的和
template <typename DataType=float>
float MatrixSum(const std::vector<std::vector<DataType>>& source_matrix);

//计算均方误差
template <typename DataType=float>
float MeanSquareError(const std::vector<std::vector<DataType>>& output_matrix, 
                      const std::vector<std::vector<DataType>>& label);

//函数定义
//模板函数 矩阵相乘(dot product)点积
template <class DataType=float>
int MatrixMultiply(const std::vector<std::vector<DataType>>& left_matrix, 
                   const std::vector<std::vector<DataType>>& right_matrix, 
                   std::vector<std::vector<DataType>>& result_matrix) {
    int flag = 0;              //判断矩阵每行的列 是否都是一个值 
    int first_cols = -1;       //第一行的列数
    int current_cols = -1;     //本行的列数
    int left_matrix_rows = -1; //左矩阵的行
    int left_matrix_cols = -1; //左矩阵的列
    int right_matrix_rows = -1;//右矩阵的行
    int right_matrix_cols = -1;//右矩阵的列

    //检查左矩阵的列
    for (int i = 0; i < left_matrix.size(); i++) {
        if (0 == i) {
            first_cols = left_matrix[i].size();
            current_cols = first_cols;
            left_matrix_cols = first_cols;
        } else {
            current_cols = left_matrix[i].size();
        }

        if (current_cols != first_cols) {
            flag = 1;
        }

        if (flag) {
            LOG(WARNING) << "矩阵相乘失败, 左矩阵每行的列数不相同...";
            return -1;
        }
    }

    //检查右矩阵的列
    for (int i = 0; i < right_matrix.size(); i++) {
        if (0 == i) {
            first_cols = right_matrix[i].size();
            current_cols = first_cols;
            right_matrix_cols = first_cols;
        } else {
            current_cols = right_matrix[i].size();
        }

        if (current_cols != first_cols) {
            flag = 1;
        }

        if (flag) {
            LOG(WARNING) << "矩阵相乘失败, 右矩阵每行的列数不相同...";
            return -1;
        }
    }
    
    //判断左矩阵的列和右矩阵的行是否相等
    left_matrix_rows = left_matrix.size();
    right_matrix_rows = right_matrix.size();
    if (left_matrix_cols != right_matrix_rows) {
        LOG(WARNING) << "矩阵相乘失败, 左矩阵的列数不等于右矩阵的行数...";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 == result_matrix.size()) {
        result_matrix = std::vector<std::vector<DataType>>(left_matrix_rows, 
                                                           std::vector<DataType>(right_matrix_cols));
    }

    //开始点积运算  
    for (int i = 0; i < left_matrix_rows; i++) {
        for (int j = 0; j < right_matrix_cols; j++) {
            for (int k = 0; k < left_matrix_cols; k++) {
                result_matrix[i][j] += left_matrix[i][k] * right_matrix[k][j];
            }
        }
    }

    return 0;
}

//hadamark积 也就是矩阵相应位置相乘
template <typename DataType=float>
int MatrixHadamarkProduct(const std::vector<std::vector<DataType>>& left_matrix, 
                          const std::vector<std::vector<DataType>>& right_matrix, 
                          std::vector<std::vector<DataType>>& result_matrix) {
    if (left_matrix.size() != right_matrix.size()) {
        LOG(WARNING) << "Hadamark积失败, 矩阵行数不同...";
        return -1;
    }
    
    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG(WARNING) << "Hadamark积失败, 矩阵列数不同...";
            return -1;
        }
    }

    if (0 == result_matrix.size()) {
        result_matrix = left_matrix;
    }

    for (int i = 0; i < left_matrix.size(); i++) {
        for (int j = 0; j < left_matrix[i].size(); j++) {
            result_matrix[i][j] = left_matrix[i][j] * right_matrix[i][j];
        }
    }

    return 0;
}

//矩阵相加
template <typename DataType=float>
int MatrixAdd(const std::vector<std::vector<DataType>>& add_matrix, 
              const std::vector<std::vector<DataType>>& beadd_matrix, 
              std::vector<std::vector<DataType>>& result_matrix) {
    if (add_matrix.size() != beadd_matrix.size()) {
        LOG(WARNING) << "矩阵相加失败, 矩阵行数不同...";
        return -1;
    }
    
    //遍历每行 是否每列都是相同的
    for (int i = 0; i < add_matrix.size(); i++) {
        if (add_matrix[i].size() != beadd_matrix[i].size()) {
            LOG(WARNING) << "矩阵相加失败, 矩阵列数不同...";
            return -1;
        }
    }

    //判断一下输出的矩阵是否是空 还没有初始化
    if (0 == result_matrix.size()) {
        result_matrix = std::vector<std::vector<DataType>>(add_matrix.size(),  
                                                           std::vector<DataType>(add_matrix[0].size()));
    }
        
    //矩阵相加
    for (int i = 0; i < add_matrix.size(); i++) {
        for (int j = 0; j < add_matrix[i].size(); j++) {
            result_matrix[i][j] = add_matrix[i][j] + beadd_matrix[i][j];
        }
    }

    return 0;
}

//矩阵相减
template <typename DataType=float>
int MatrixSubtract(const std::vector<std::vector<DataType>>& sub_matrix, 
                   const std::vector<std::vector<DataType>>& besub_matrix, 
                   std::vector<std::vector<DataType>>& result_matrix) {
    if (sub_matrix.size() != besub_matrix.size()) {
        LOG(WARNING) << "矩阵相减失败, 矩阵行数不同...";
        return -1;
    }
    
    //遍历每行 是否每列都是相同的
    for (int i = 0; i < sub_matrix.size(); i++) {
        if (sub_matrix[i].size() != besub_matrix[i].size()) {
            LOG(WARNING) << "矩阵相加失败, 矩阵列数不同...";
            return -1;
        }
    }

    //判断一下输出的矩阵是否是空 还没有初始化
    if (0 == result_matrix.size()) {
        result_matrix = std::vector<std::vector<DataType>>(sub_matrix.size(),  
                                                           std::vector<DataType>(sub_matrix[0].size()));
    }
        
    //矩阵相减
    for (int i = 0; i < sub_matrix.size(); i++) {
        for (int j = 0; j < sub_matrix[i].size(); j++) {
            result_matrix[i][j] = sub_matrix[i][j] - besub_matrix[i][j];
        }
    }

    return 0;
}

//matrix reshape  把原矩阵变成 几行几列
template <typename DataType=float>
int MatrixReshape(const std::vector<std::vector<DataType>>& source_matrix,
                  size_t rows, size_t cols, 
                  std::vector<std::vector<DataType>>& result_matrix) {
    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = 0;
    for (int i = 0; i < source_matrix.size(); i++) {
        matrix_total_size += source_matrix[i].size();
    }
    
    if (matrix_total_size != (rows * cols)) {
        LOG(WARNING) << "输入矩阵不能reshape成指定形状";
        return -1;
    }

    //先把值放入一个一维数组
    std::vector<DataType> matrix_data(matrix_total_size);
    int index = 0;
    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            matrix_data[index++] = source_matrix[i][j];
        }
    }
    
    index = 0;
    //再赋值给新数组  判断一下输出矩阵有没有初始化
    if (0 == result_matrix.size()) {
        result_matrix.reserve(rows);

        for (int i = 0; i < rows; i++) {
            std::vector<DataType> cols_array;
            cols_array.reserve(cols);
            for (int j = 0; j < cols; j++) {
                cols_array.push_back(matrix_data[index++]);
            }
            result_matrix.push_back(cols_array);
        }
    } else {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_matrix[i][j] = matrix_data[index++];    
            }
        }
    }

    return 0;
}

//接收输入矩阵为1维矩阵 函数重载
template <typename DataType=float>
int MatrixReshape(const std::vector<DataType>& source_matrix,
                  size_t rows, size_t cols, 
                  std::vector<std::vector<DataType>>& result_matrix) { 
    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = source_matrix.size();
    
    if (matrix_total_size != (rows * cols)) {
        LOG(WARNING) << "输入矩阵不能reshape成指定形状";
        return -1;
    }
    
    //判断输出矩阵有没有初始化
    if (0 == result_matrix.size()) {
        result_matrix = std::vector<std::vector<DataType>>(rows, 
                                                           std::vector<DataType>(cols));
    }

    int index = 0;
    //再赋值给新数组
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = source_matrix[index++];    
        }
    }

    return 0;
}

//打印矩阵
template <typename DataType=float>
void MatrixShow(const std::vector<std::vector<DataType>>& matrix) {
    if (0 == matrix.size()) {
        LOG(WARNING) << "矩阵为空...";
        return ;
    }

    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            if (0 == j) {
                std::cout << "  [";
            }

            //如果是负数 则会多占一格 那么是正数 就多加个空格
            //设置浮点的格式 后面6位
            int space_number = 0;
            if (matrix[i][j] >= 0) {
                if ((matrix[i][j] / 10.0) < 1.0) {
                    space_number = 3;
                } else {
                    space_number = 2;
                }
            } else {
                if ((matrix[i][j] / 10.0) < 1.0) {
                    space_number = 2;
                } else {
                    space_number = 1;
                }
            }

            std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                      << std::setprecision(6) << std::string(space_number, ' ')
                      << matrix[i][j];
            
            if ((j + 1) == matrix[i].size()) {
                std::cout << "  ]" << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

//创建0矩阵
template <typename DataType=float>
void CreateZerosMatrix(int rows, int cols, 
                       std::vector<std::vector<DataType>>& matrix) {
    if (0 != matrix.size()) {
        matrix.clear();
    }
    
    matrix.reserve(rows);

    for (int i = 0; i < rows; i++) {
        std::vector<DataType> cols_array(cols);
        for (int j = 0; j < cols; j++) {
            cols_array[j] = 0;
        }
        matrix.push_back(cols_array);
    }
}

//创建1矩阵
template <typename DataType=float>
void CreateOnesMatrix(int rows, int cols, 
                      std::vector<std::vector<DataType>>& matrix) {
    if (0 != matrix.size()) {
        matrix.clear();
    }

    matrix.reserve(rows);

    for (int i = 0; i < rows; i++) {
        std::vector<DataType> cols_array(cols);
        for (int j = 0; j < cols; j++) {
            cols_array[j] = 1;
        }
        matrix.push_back(cols_array);
    }
}

//转置矩阵
template <typename DataType=float>
int TransposeMatrix(const std::vector<std::vector<DataType>>& source_matrix, 
                    std::vector<std::vector<DataType>>& result_matrix) {
    int source_matrix_cols = 0;
    int source_matrix_rows = source_matrix.size();
    //检查源矩阵每列是否相同
    for (int i = 0; i < source_matrix.size(); i++) {
        if (0 == i) {
           source_matrix_cols = source_matrix[i].size();
        } else {
            if (source_matrix_cols != source_matrix[i].size()) {
               LOG(WARNING) << "输入矩阵每行的列数不相同...";
               return -1;
            }
        }
    }
    
    //如果数组数据没有初始化 就用移动赋值函数初始化 
    //行为原矩阵的列 列为原矩阵的行 比如2 * 4  变成4 * 2
    if (0 == result_matrix.size()) {
        result_matrix = std::vector<std::vector<DataType>>(source_matrix_cols, 
                                                           std::vector<DataType>(source_matrix_rows));
    }

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            result_matrix[j][i] = source_matrix[i][j];
        }
    }

    return 0;
}

//矩阵都乘以一个值
template <typename DataType=float>
void ValueMulMatrix(DataType value,  
                    const std::vector<std::vector<DataType>>& source_matrix, 
                    std::vector<std::vector<DataType>>& result_matrix) {
    if (0 == result_matrix.size()) {
        result_matrix = source_matrix;
    }

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            result_matrix[i][j] = source_matrix[i][j] * value;
        }
    }
}

//一个值减去矩阵每个值
template <typename DataType=float>
void ValueSubMatrix(DataType value,  
                    const std::vector<std::vector<DataType>>& source_matrix, 
                    std::vector<std::vector<DataType>>& result_matrix) {
    if (0 == result_matrix.size()) {
        result_matrix = source_matrix;
    }

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            result_matrix[i][j] = value - source_matrix[i][j];
        }
    }
}

//计算2d矩阵的和
template <typename DataType=float>
float MatrixSum(const std::vector<std::vector<DataType>>& source_matrix) {
    float sum = 0.0;
    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            sum += source_matrix[i][j];
        }
    }

    return sum;
}

//均方误差 ((y - predict)**2.sum()) / 2 
template <typename DataType=float>
float MeanSquareError(const std::vector<std::vector<DataType>>& output_matrix, 
                      const std::vector<std::vector<DataType>>& label) {
    if (output_matrix.size() != label.size()) {
        LOG(WARNING) << "计算均方误差失败, 矩阵的行数不相同...";
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
            LOG(WARNING) << "计算均方误差失败, 矩阵的列数不相同...";
        }
    }

    //计算均方误差
    float sum = 0.0;
    for (int i = 0; i < output_matrix.size(); i++) {
        for (int j = 0; j < output_matrix[i].size(); j++) {
             sum += pow((label[i][j] - output_matrix[i][j]), 2);
        }
    }

    return sum / 2;
}

}     //namespace matrix

//随机数生成
namespace random {

//函数声明
//生成正态分布的随机数二维矩阵`
template <typename DataType=float>
int Uniform(float a, float b, int rows, int cols, 
            std::vector<std::vector<DataType>>& random_matrix);

template <typename DataType=float>
int Random(int a, int b, int rows, int cols, 
           std::vector<std::vector<DataType>>& random_matrix);

//生成一个 rows * cols的随机数矩阵 值的范围在a 到 b之间 
template <typename DataType=float>
int Uniform(float a, float b, int rows, int cols, 
            std::vector<std::vector<DataType>>& random_matrix) {
    if (b < a) {
        LOG(WARNING) << "随机数生成错误 下限大于上限";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    float mean = a + ((b - a) / 2);   //均值
    float stddev = (b - a) / 2;       //标准差

    std::default_random_engine generate_engine;                      //生成引擎
    std::normal_distribution<DataType> generate_random(mean, stddev);//标准正态分布 
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(generate_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}

template <typename DataType=float>
int Random(int a, int b, int rows, int cols,  
           std::vector<std::vector<DataType>>& random_matrix) {
    if (b < a) {
        LOG(WARNING) << "随机数生成错误 下限大于上限";
        return -1;
    }

    if (0 == random_matrix.size()) {
        matrix::CreateZerosMatrix(rows, cols, random_matrix);
    }

    DataType random_value = 0;
    //用time函数的返回值 来做seed种子
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            srand((unsigned int)time(NULL) + i * cols + j * 100000);
            if (a >= 0) {
                random_value = rand() % b + a;
                random_value > b ? b : random_value;
            } else {
                random_value = rand() % (b - a) + a;
            }
            random_matrix[i][j] = random_value;
        }
    }

    return 0;
}


}         //namespace random
}         //namespace calculate

#endif    //MATRIX_MATH_FUNCTIONS_HPP_
