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
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef MATRIX_MATH_FUNCTIONS_HPP_
#define MATRIX_MATH_FUNCTIONS_HPP_

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

#include <glog/logging.h>

namespace calculate {

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
        
    //矩阵相加
    for (int i = 0; i < add_matrix.size(); i++) {
        for (int j = 0; j < add_matrix[i].size(); j++) {
            result_matrix[i][j] = add_matrix[i][j] + beadd_matrix[i][j];
        }
    }

    return 0;
}

//matrix reshape  把原矩阵变成 几行几列
template <typename DataType=float>
int MatrixReshape(const std::vector<std::vector<DataType>>& source_matrix,
                  std::vector<std::vector<DataType>>& result_matrix, 
                  size_t rows, size_t cols) {
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
    //再赋值给新数组
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = matrix_data[index++];    
        }
    }

    return 0;
}

//接收输入矩阵为1维矩阵 函数重载
template <typename DataType=float>
int MatrixReshape(const std::vector<DataType>& source_matrix,
                  std::vector<std::vector<DataType>>& result_matrix, 
                  size_t rows, size_t cols) {
    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = source_matrix.size();
    
    if (matrix_total_size != (rows * cols)) {
        LOG(WARNING) << "输入矩阵不能reshape成指定形状";
        return -1;
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
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            if (0 == j) {
                std::cout << "  [";
           }
            std::cout << std::showpoint << "  " << matrix[i][j];
            
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
    matrix.reserve(rows);

    for (int i = 0; i < rows; i++) {
        std::vector<DataType> cols_array(cols);
        for (int j = 0; j < cols; j++) {
            cols_array[j] = 1;
        }
        matrix.push_back(cols_array);
    }
}

//随机数生成
namespace random {

//生成一个 rows * cols的随机数矩阵 值的范围在a 到 b之间
template <typename DataType=float>
int Uniform(float a, float b, int rows, int cols, 
            std::vector<std::vector<DataType>>& random_matrix) {
    if (b < a) {
        LOG(WARNING) << "随机数生成错误 下限大于上限";
        return -1;
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

}         //namespace random
}         //namespace calculate

#endif    //MATRIX_MATH_FUNCTIONS_HPP_
