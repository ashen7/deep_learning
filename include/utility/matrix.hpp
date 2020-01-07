/*
 * =====================================================================================
 *
 *       Filename:  matrix.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月04日 16时42分49秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef CALCULATE_MATRIX_HPP_
#define CALCULATE_MATRIX_HPP_

#include <time.h>
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
#include <chrono>

#include <glog/logging.h>

//模板类 
//模板类有一层命名空间  
namespace calculate {
namespace matrix {

template <typename DataType=double>
struct Matrix {
    //类型别名
    typedef std::vector<DataType> Matrix1d;
    typedef std::vector<std::vector<DataType>> Matrix2d;
    typedef std::vector<std::vector<std::vector<DataType>>> Matrix3d;
    
    // 矩阵相乘  dot product 点积
    static int MatrixDotProduct(const Matrix2d& left_matrix, 
                                const Matrix2d& right_matrix, 
                                Matrix2d& result_matrix); 

    // 矩阵相乘 函数重载 输入为uint8_t类型 输出为double类型
    static int MatrixDotProduct(const std::vector<std::vector<double>>& left_matrix, 
                                const std::vector<std::vector<uint8_t>>& right_matrix, 
                                std::vector<std::vector<double>>& result_matrix); 

    // 矩阵对应位置相乘 hadamark积 
    static int MatrixHadamarkProduct(const Matrix2d& left_matrix, 
                                     const Matrix2d& right_matrix, 
                                     Matrix2d& result_matrix);

    // 矩阵相加 
    static int MatrixAdd(const Matrix2d& add_matrix, 
                         const Matrix2d& beadd_matrix, 
                         Matrix2d& result_matrix);

    // 矩阵相减
    static int MatrixSubtract(const Matrix2d& sub_matrix, 
                              const Matrix2d& besub_matrix, 
                              Matrix2d& result_matrix);

    // 返回元祖 矩阵的形状(宽, 高)
    static std::tuple<size_t, size_t> GetMatrixShape(const Matrix2d& source_matrix);

    // 矩阵reshape 成几行几列  这是二维矩阵版
    static int MatrixReshape(const Matrix2d& source_matrix,
                             size_t rows, size_t cols, 
                             Matrix2d& result_matrix);

    // 矩阵reshape 一维矩阵转二维矩阵版
    static int MatrixReshape(const Matrix1d& source_matrix,
                             size_t rows, size_t cols, 
                             Matrix2d& result_matrix);

    // 打印二维矩阵
    static void MatrixShow(const Matrix2d& matrix);

    // 打印图像二维矩阵
    static void ImageMatrixShow(const std::vector<std::vector<uint8_t>>& matrix);

    // 创建2维矩阵 初始值为0
    static int CreateZerosMatrix(size_t rows, size_t cols, 
                                 Matrix2d& matrix);

    // 创建2维矩阵 初始值为0
    static int CreateZerosMatrix(const std::tuple<size_t, size_t>& shape, 
                                 Matrix2d& matrix);

    // 创建2维矩阵 初始值为1
    static int CreateOnesMatrix(size_t rows, size_t cols, 
                                Matrix2d& matrix);

    // 创建2维矩阵 初始值为1
    static int CreateOnesMatrix(const std::tuple<size_t, size_t>& shape, 
                                Matrix2d& matrix);

    // 2维矩阵的装置矩阵
    static int TransposeMatrix(const Matrix2d& source_matrix, 
                               Matrix2d& result_matrix);

    // 1个值 乘以 一个2d矩阵
    static void ValueMulMatrix(DataType value,  
                               const Matrix2d& source_matrix, 
                               Matrix2d& result_matrix);

    // 1个值 减去 一个2d矩阵
    static void ValueSubMatrix(DataType value,  
                               const Matrix2d& source_matrix, 
                               Matrix2d& result_matrix);

    //计算2d矩阵的和
    static double Sum(const Matrix2d& source_matrix);

    //计算均方误差
    static double MeanSquareError(const Matrix2d& output_matrix, 
                                  const Matrix2d& label);

    //判断矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<double>>& Matrix) {
        return false;
    }

    //判断矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<uint8_t>>& Matrix) {
        return true;
    }

};    //struct Matrix 


//矩阵相乘(dot product)点积
template <class DataType>
int Matrix<DataType>::MatrixDotProduct(const Matrix2d& left_matrix, 
                                       const Matrix2d& right_matrix, 
                                       Matrix2d& result_matrix) {
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
            LOG(ERROR) << "matrix dot product failed, left matrix cols of each line is not equal";
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
            LOG(ERROR) << "matrix dot product failed, right matrix cols of each line is not equal";
            return -1;
        }
    }
    
    //判断左矩阵的列和右矩阵的行是否相等
    left_matrix_rows = left_matrix.size();
    right_matrix_rows = right_matrix.size();
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //判断输出矩阵是否初始化过 是的话要清零
    if (0 != result_matrix.size()) {
        result_matrix.clear();
    }
    result_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

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

//矩阵相乘函数重载
template <class DataType>
int Matrix<DataType>::MatrixDotProduct(const std::vector<std::vector<double>>& left_matrix, 
                                       const std::vector<std::vector<uint8_t>>& right_matrix, 
                                       std::vector<std::vector<double>>& result_matrix) {
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
            LOG(ERROR) << "matrix dot product failed, left matrix cols of each line is not equal";
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
            LOG(ERROR) << "matrix dot product failed, right matrix cols of each line is not equal";
            return -1;
        }
    }
    
    //判断左矩阵的列和右矩阵的行是否相等
    left_matrix_rows = left_matrix.size();
    right_matrix_rows = right_matrix.size();
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //判断输出矩阵是否初始化过 是的话要清零
    if (0 != result_matrix.size()) {
        result_matrix.clear();
    }
    result_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

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
template <typename DataType>
int Matrix<DataType>::MatrixHadamarkProduct(const Matrix2d& left_matrix, 
                                            const Matrix2d& right_matrix, 
                                            Matrix2d& result_matrix) {
    if (left_matrix.size() != right_matrix.size()) {
        LOG(ERROR) << "matrix hadamark product failed, left matrix rows is not equal of right matrix rows";
        return -1;
    }
    
    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG(ERROR) << "matrix hadamark product failed, left matrix cols is not equal of right matrix cols";
            return -1;
        }
    }

    if (0 == result_matrix.size()) {
        result_matrix = left_matrix;
    } else {
        //对输出矩阵做一下检查 看行列是否符合 如果不对就清空容器 重新赋值
        if (result_matrix.size() != left_matrix.size()) {
            result_matrix.clear();
            result_matrix = left_matrix;
        } else {
            //如果行符合了 再来检查一下列
            for (int i = 0; i < result_matrix.size(); i++) {
                if (result_matrix[i].size() != left_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = left_matrix;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < left_matrix.size(); i++) {
        for (int j = 0; j < left_matrix[i].size(); j++) {
            result_matrix[i][j] = left_matrix[i][j] * right_matrix[i][j];
        }
    }

    return 0;
}

//矩阵相加
template <typename DataType>
int Matrix<DataType>::MatrixAdd(const Matrix2d& add_matrix, 
                                const Matrix2d& beadd_matrix, 
                                Matrix2d& result_matrix) {
    if (add_matrix.size() != beadd_matrix.size()) {
        LOG(ERROR) << "matrix add failed, left matrix rows is not equal of right matrix rows";
        return -1;
    }
    
    //遍历每行 是否每列都是相同的
    for (int i = 0; i < add_matrix.size(); i++) {
        if (add_matrix[i].size() != beadd_matrix[i].size()) {
            LOG(ERROR) << "matrix add failed, left matrix cols is not equal of right matrix cols";
            return -1;
        }
    }

    //判断一下输出的矩阵是否是空 还没有初始化
    if (0 == result_matrix.size()) {
        result_matrix = Matrix2d(add_matrix.size(), Matrix1d(add_matrix[0].size()));
    } else {
        //对输出矩阵做一下检查 看行列是否符合 如果不对就清空容器 重新赋值
        if (result_matrix.size() != add_matrix.size()) {
            result_matrix.clear();
            result_matrix = add_matrix;
        } else {
            //如果行符合了 再来检查一下列
            for (int i = 0; i < result_matrix.size(); i++) {
                if (result_matrix[i].size() != add_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = add_matrix;
                    break;
                }
            }
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

//矩阵相减
template <typename DataType>
int Matrix<DataType>::MatrixSubtract(const Matrix2d& sub_matrix, 
                                     const Matrix2d& besub_matrix, 
                                     Matrix2d& result_matrix) {
    if (sub_matrix.size() != besub_matrix.size()) {
        LOG(ERROR) << "matrix subtract failed, left matrix rows is not equal of right matrix rows";
        return -1;
    }
    
    //遍历每行 是否每列都是相同的
    for (int i = 0; i < sub_matrix.size(); i++) {
        if (sub_matrix[i].size() != besub_matrix[i].size()) {
            LOG(ERROR) << "matrix subtract failed, left matrix cols is not equal of right matrix cols";
            return -1;
        }
    }

    //判断一下输出的矩阵是否是空 还没有初始化
    if (0 == result_matrix.size()) {
        result_matrix = Matrix2d(sub_matrix.size(), Matrix1d(sub_matrix[0].size()));
    } else {
        //对输出矩阵做一下检查 看行列是否符合 如果不对就清空容器 重新赋值
        if (result_matrix.size() != sub_matrix.size()) {
            result_matrix.clear();
            result_matrix = sub_matrix;
        } else {
            //如果行符合了 再来检查一下列
            for (int i = 0; i < result_matrix.size(); i++) {
                if (result_matrix[i].size() != sub_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = sub_matrix;
                    break;
                }
            }
        }
    }
        
    //矩阵相减
    for (int i = 0; i < sub_matrix.size(); i++) {
        for (int j = 0; j < sub_matrix[i].size(); j++) {
            result_matrix[i][j] = sub_matrix[i][j] - besub_matrix[i][j];
        }
    }

    return 0;
}

//得到矩阵的形状
template <typename DataType>
std::tuple<size_t, size_t> Matrix<DataType>::GetMatrixShape(const Matrix2d& source_matrix) {
    if (0 == source_matrix.size()) {
        LOG(ERROR) << "get matrix shape failed, input matrix is empty";
        return std::make_tuple(0, 0);
    }
    
    size_t height = source_matrix.size();
    size_t width = source_matrix[0].size();
    
    for (int i = 0; i < source_matrix.size(); i++) {
        if (width != source_matrix[i].size()) {
            LOG(ERROR) << "get matrix shape failed, input matrix cols of each line is not equal";
            return std::make_tuple(0, 0);
        }
    }

    return std::make_tuple(height, width);
}

//matrix reshape  把原矩阵变成 几行几列
template <typename DataType>
int Matrix<DataType>::MatrixReshape(const Matrix2d& source_matrix,
                                    size_t rows, size_t cols, 
                                    Matrix2d& result_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }
    
    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = 0;
    for (int i = 0; i < source_matrix.size(); i++) {
        matrix_total_size += source_matrix[i].size();
    }
    
    if (matrix_total_size != (rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }

    //先把值放入一个一维数组
    Matrix1d matrix_data(matrix_total_size);
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
            Matrix1d cols_array;
            cols_array.reserve(cols);
            for (int j = 0; j < cols; j++) {
                cols_array.push_back(matrix_data[index++]);
            }
            result_matrix.push_back(cols_array);
        }
    } else {
        //对输出矩阵的行列做判断
        if (rows != result_matrix.size()) {
            result_matrix.clear();
            result_matrix = source_matrix;
        } else {
            for (int i = 0; i < result_matrix.size(); i++) {
                if (cols != result_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = source_matrix;
                    break;
                }
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_matrix[i][j] = matrix_data[index++];    
            }
        }
    }

    return 0;
}

//接收输入矩阵为1维矩阵 函数重载
template <typename DataType>
int Matrix<DataType>::MatrixReshape(const Matrix1d& source_matrix,
                                    size_t rows, size_t cols, 
                                    Matrix2d& result_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = source_matrix.size();
    
    if (matrix_total_size != (rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }
    
    //判断输出矩阵有没有初始化
    if (0 == result_matrix.size()) {
        result_matrix = Matrix2d(rows, Matrix1d(cols));
    } else {
        //对输出矩阵的行列做判断
        if (rows != result_matrix.size()) {
            result_matrix.clear();
            result_matrix = Matrix2d(rows, Matrix1d(cols));
        } else {
            for (int i = 0; i < result_matrix.size(); i++) {
                if (cols != result_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = Matrix2d(rows, Matrix1d(cols));
                    break;
                }
            }
        }
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
template <typename DataType>
void Matrix<DataType>::MatrixShow(const Matrix2d& matrix) {
    if (0 == matrix.size()) {
        LOG(ERROR) << "print matrix failed, input matrix is empty";
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
                      << std::setprecision(8) << std::string(space_number, ' ')
                      << matrix[i][j];
            
            if ((j + 1) == matrix[i].size()) {
                std::cout << "  ]" << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

// 打印图像二维矩阵
template <typename DataType>
void Matrix<DataType>::ImageMatrixShow(const std::vector<std::vector<uint8_t>>& matrix) {
    if (0 == matrix.size()) {
        LOG(ERROR) << "print matrix failed, input matrix is empty";
        return ;
    }

    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            if (0 == j) {
                std::cout << "  [";
            }

            std::cout << std::setw(3) << std::setiosflags(std::ios::right)
                      << int(matrix[i][j]) << " ";
            
            if ((j + 1) == matrix[i].size()) {
                std::cout << " ]" << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

//创建0矩阵
template <typename DataType>
int Matrix<DataType>::CreateZerosMatrix(size_t rows, size_t cols, 
                                        Matrix2d& matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    if (0 != matrix.size()) {
        matrix.clear();
    }
    
    matrix.reserve(rows);

    for (int i = 0; i < rows; i++) {
        Matrix1d cols_array(cols);
        for (int j = 0; j < cols; j++) {
            cols_array[j] = 0;
        }
        matrix.push_back(cols_array);
    }
}

//创建0矩阵
template <typename DataType>
int Matrix<DataType>::CreateZerosMatrix(const std::tuple<size_t, size_t>& shape, 
                                        Matrix2d& matrix) {
    size_t rows = -1;
    size_t cols = -1;
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    if (0 != matrix.size()) {
        matrix.clear();
    }
    
    matrix.reserve(rows);

    for (int i = 0; i < rows; i++) {
        Matrix1d cols_array(cols);
        for (int j = 0; j < cols; j++) {
            cols_array[j] = 0;
        }
        matrix.push_back(cols_array);
    }
}

//创建1矩阵
template <typename DataType>
int Matrix<DataType>::CreateOnesMatrix(size_t rows, size_t cols, 
                                       Matrix2d& matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    if (0 != matrix.size()) {
        matrix.clear();
    }

    matrix.reserve(rows);

    for (int i = 0; i < rows; i++) {
        Matrix1d cols_array(cols);
        for (int j = 0; j < cols; j++) {
            cols_array[j] = 1;
        }
        matrix.push_back(cols_array);
    }
}

//创建1矩阵
template <typename DataType>
int Matrix<DataType>::CreateOnesMatrix(const std::tuple<size_t, size_t>& shape, 
                                       Matrix2d& matrix) {
    size_t rows = -1;
    size_t cols = -1;
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    if (0 != matrix.size()) {
        matrix.clear();
    }

    matrix.reserve(rows);

    for (int i = 0; i < rows; i++) {
        Matrix1d cols_array(cols);
        for (int j = 0; j < cols; j++) {
            cols_array[j] = 1;
        }
        matrix.push_back(cols_array);
    }
}

//转置矩阵
template <typename DataType>
int Matrix<DataType>::TransposeMatrix(const Matrix2d& source_matrix, 
                                      Matrix2d& result_matrix) {
    int source_matrix_cols = 0;
    int source_matrix_rows = source_matrix.size();
    //检查源矩阵每列是否相同
    for (int i = 0; i < source_matrix.size(); i++) {
        if (0 == i) {
           source_matrix_cols = source_matrix[i].size();
        } else {
            if (source_matrix_cols != source_matrix[i].size()) {
                LOG(ERROR) << "transpose matrix failed, input matrix cols of each line is not equal";
               return -1;
            }
        }
    }
    
    //如果数组数据没有初始化 就用移动赋值函数初始化 
    //行为原矩阵的列 列为原矩阵的行 比如2 * 4  变成4 * 2
    if (0 == result_matrix.size()) {
        result_matrix = Matrix2d(source_matrix_cols, Matrix1d(source_matrix_rows));
    } else {
        //对输出矩阵的行列做判断
        if (source_matrix_cols != result_matrix.size()) {
            result_matrix.clear();
            result_matrix = Matrix2d(source_matrix_cols, Matrix1d(source_matrix_rows));
        } else {
            for (int i = 0; i < result_matrix.size(); i++) {
                if (source_matrix_rows != result_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = Matrix2d(source_matrix_cols, Matrix1d(source_matrix_rows));
                    break;
                }
            }
        }
    }

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            result_matrix[j][i] = source_matrix[i][j];
        }
    }

    return 0;
}

//矩阵都乘以一个值
template <typename DataType>
void Matrix<DataType>::ValueMulMatrix(DataType value,  
                                      const Matrix2d& source_matrix, 
                                      Matrix2d& result_matrix) {
    if (0 == result_matrix.size()) {
        result_matrix = source_matrix;
    } else {
        //对输出矩阵的行列做判断
        if (source_matrix.size() != result_matrix.size()) {
            result_matrix.clear();
            result_matrix = source_matrix;
        } else {
            for (int i = 0; i < result_matrix.size(); i++) {
                if (source_matrix[i].size() != result_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = source_matrix;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            result_matrix[i][j] = source_matrix[i][j] * value;
        }
    }
}

//一个值减去矩阵每个值
template <typename DataType>
void Matrix<DataType>::ValueSubMatrix(DataType value,  
                                      const Matrix2d& source_matrix, 
                                      Matrix2d& result_matrix) {
    if (0 == result_matrix.size()) {
        result_matrix = source_matrix;
    } else {
        //对输出矩阵的行列做判断
        if (source_matrix.size() != result_matrix.size()) {
            result_matrix.clear();
            result_matrix = source_matrix;
        } else {
            for (int i = 0; i < result_matrix.size(); i++) {
                if (source_matrix[i].size() != result_matrix[i].size()) {
                    result_matrix.clear();
                    result_matrix = source_matrix;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            result_matrix[i][j] = value - source_matrix[i][j];
        }
    }
}

//计算2d矩阵的和
template <typename DataType>
double Matrix<DataType>::Sum(const Matrix2d& source_matrix) {
    double sum = 0.0;
    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            sum += source_matrix[i][j];
        }
    }

    return sum;
}

//均方误差 ((y - predict)**2.sum()) / 2 
template <typename DataType>
double Matrix<DataType>::MeanSquareError(const Matrix2d& output_matrix, 
                                        const Matrix2d& label) {
    if (output_matrix.size() != label.size()) {
        LOG(ERROR) << "calculate mean square error failed, two matrices rows is not equal";
        return -1;
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
            LOG(ERROR) << "calculate mean square error failed, two matrices cols is not equal";
            return -1;
        }
    }

    //计算均方误差
    double sum = 0.0;
    for (int i = 0; i < output_matrix.size(); i++) {
        for (int j = 0; j < output_matrix[i].size(); j++) {
             sum += pow((label[i][j] - output_matrix[i][j]), 2);
        }
    }

    return sum / 2;
}




}       //namespace matrix




namespace random {

//模板类 随机数对象
template <typename DataType=double>
struct Random { 
    //类型别名
    typedef std::vector<DataType> Matrix1d;
    typedef std::vector<std::vector<DataType>> Matrix2d;
    typedef std::vector<std::vector<std::vector<DataType>>> Matrix3d;

    //生成服从正态分布的随机数二维矩阵
    static int Normal(float mean, float stddev, size_t rows, size_t cols, 
                      std::vector<std::vector<double>>& random_matrix);

    //生成随机的浮点数二维矩阵
    static int Uniform(float a, float b, size_t rows, size_t cols, 
                       std::vector<std::vector<double>>& random_matrix);

    //生成随机的整数二维矩阵
    static int RandInt(float a, float b, size_t rows, size_t cols, 
                       Matrix2d& random_matrix);


};   //struct Random

//生成服从正态分布的随机数二维矩阵
template <typename DataType>
int Random<DataType>::Normal(float mean, float stddev, size_t rows, size_t cols, 
                             std::vector<std::vector<double>>& random_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "get normal distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get normal distribution matrix failed, input cols <= 0";
        return -1;
    }

    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::normal_distribution<double> generate_random(mean, stddev);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }
}


//生成一个 rows * cols的随机数矩阵 值的范围在a 到 b之间 
template <typename DataType>
int Random<DataType>::Uniform(float a, float b, size_t rows, size_t cols, 
                              std::vector<std::vector<double>>& random_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input cols <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_real_distribution<double> generate_random(a, b);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}

//生成一个随机整数二维矩阵
template <typename DataType>
int Random<DataType>::RandInt(float a, float b, size_t rows, size_t cols,  
                              Matrix2d& random_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input cols <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_int_distribution<int> generate_random(a, b);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}


}       //namespace random





namespace activator {

//模板类  激活函数
template <typename DataType=double>
struct Activator {
    //类型别名
    typedef std::vector<double> Matrix1d;
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;

    //sigmoid激活函数的前向计算
    static void SigmoidForward(const Matrix2d& input_array, 
                               Matrix2d& output_array);

    //sigmoid激活函数的反向计算
    static void SigmoidBackward(const Matrix2d& output_array, 
                                Matrix2d& delta_array);

    //sigmoid激活函数的反向计算
    static void SigmoidBackward(const ImageMatrix2d& output_array, 
                                Matrix2d& delta_array);

};        //struct Activator


//sigmoid激活函数的前向计算
template <typename DataType>
void Activator<DataType>::SigmoidForward(const Matrix2d& input_array, 
                                         Matrix2d& output_array) { 
    //如果输出数组未初始化 
    if (0 == output_array.size()) {
        output_array = input_array;
    }

    //计算 1 / (1 + exp(-input_array))
    for (int i = 0; i < input_array.size(); i++) {
        for (int j = 0; j < input_array[i].size(); j++) {
            //exp返回e的x次方 得到0. 1. 2.值 加上1都大于1了 然后用1除  最后都小于1
            output_array[i][j] = 1.0 / (1.0 + exp(-input_array[i][j])); 
        }
    }
}

//sigmoid激活函数的反向计算
template <typename DataType>
void Activator<DataType>::SigmoidBackward(const Matrix2d& output_array, 
                                          Matrix2d& delta_array) {
    //如果输出数组未初始化 
    if (0 == delta_array.size()) {
        delta_array = output_array;
    }

    //计算 output(1 - output)
    for (int i = 0; i < output_array.size(); i++) {
        for (int j = 0; j < output_array[i].size(); j++) {
            delta_array[i][j] = output_array[i][j] * (1.0 - output_array[i][j]);
        }
    }
}

//sigmoid激活函数的反向计算
template <typename DataType>
void Activator<DataType>::SigmoidBackward(const ImageMatrix2d& output_array, 
                                          Matrix2d& delta_array) {
    //如果输出数组未初始化 
    if (0 == delta_array.size()) {
        auto shape = matrix::Matrix<uint8_t>::GetMatrixShape(output_array);
        matrix::Matrix<double>::CreateZerosMatrix(shape, delta_array);
    }

    //计算 output(1 - output)
    for (int i = 0; i < output_array.size(); i++) {
        for (int j = 0; j < output_array[i].size(); j++) {
            delta_array[i][j] = output_array[i][j] * (1.0 - output_array[i][j]);
        }
    }
}




}         //namespace activator






namespace time {
static void GetCurrentTime(char* now_time);


void GetCurrentTime(char* now_time) {
    time_t now = std::chrono::system_clock::to_time_t(
                              std::chrono::system_clock::now());
    
    struct tm* ptime = localtime(&now);
    sprintf(now_time, "%d-%02d-%02d %02d:%02d:%02d",
		    (int)ptime->tm_year + 1900, (int)ptime->tm_mon + 1, 
            (int)ptime->tm_mday, (int)ptime->tm_hour, 
            (int)ptime->tm_min, (int)ptime->tm_sec);
}

}         //namespace time
}         //namespace calculate


//定义别名
typedef calculate::matrix::Matrix<double> Matrix;
typedef calculate::random::Random<double> Random;
typedef calculate::activator::Activator<double> Activator;

//图像矩阵
typedef calculate::matrix::Matrix<uint8_t> ImageMatrix;
typedef calculate::random::Random<uint8_t> ImageRandom;
typedef calculate::activator::Activator<uint8_t> ImageActivator;


#endif    //CALCULATE_MATRIX_HPP_
