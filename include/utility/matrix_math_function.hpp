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
#include <omp.h>

#define OPENMP_THREADS_NUMBER 6   //openmp并行线程数量

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
    
    // 检查2d矩阵行列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const Matrix2d& matrix, bool is_write_logging);

    // 检查3d矩阵深度 行 列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const Matrix3d& matrix, bool is_write_logging);

    // 检查两个2d矩阵行 列是否正确
    static bool MatrixCheck(const Matrix2d& left_matrix, 
                            const Matrix2d& right_matrix, 
                            bool is_write_logging);

    // 检查两个3d矩阵 深度 行 列是否正确
    static bool MatrixCheck(const Matrix3d& left_matrix, 
                            const Matrix3d& right_matrix, 
                            bool is_write_logging);

    // 检查2d矩阵行 列 是否是 期望的 行 列  
    static bool MatrixCheck(const Matrix2d& matrix, 
                            int32_t rows, int32_t cols, 
                            bool is_write_logging);

    // 检查3d矩阵深度 行 列 是否是 期望的 深度 行 列  
    static bool MatrixCheck(const Matrix3d& matrix, int32_t depth, 
                            int32_t height, int32_t width, 
                            bool is_write_logging);

    // 返回元祖 2d矩阵的形状(高, 宽)
    static std::tuple<int32_t, int32_t> GetShape(const Matrix2d& source_matrix);

    // 返回元祖 3d矩阵的形状(深度, 高, 宽)
    static std::tuple<int32_t, int32_t, int32_t> GetShape(const Matrix3d& source_matrix);

    // 返回一个浮点型的2d矩阵
    static std::vector<std::vector<double>> ToDouble(const std::vector<std::vector<uint8_t>>& matrix);

    // 返回一个浮点型的3d矩阵
    static std::vector<std::vector<std::vector<double>>> ToDouble(const std::vector<std::vector<std::vector<uint8_t>>>& matrix);

    // 创建2维矩阵 初始值为0
    static int8_t CreateZeros(int32_t rows, int32_t cols, 
                              Matrix2d& matrix);

    // 创建2维矩阵 初始值为0
    static int8_t CreateZeros(const std::tuple<int32_t, int32_t>& shape, 
                              Matrix2d& matrix);

    // 创建3维矩阵 初始值为0
    static int8_t CreateZeros(int32_t depth, int32_t height, int32_t width, 
                              Matrix3d& matrix);

    // 创建3维矩阵 初始值为0
    static int8_t CreateZeros(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                              Matrix3d& matrix);

    // 创建2维矩阵 初始值为1
    static int8_t CreateOnes(int32_t rows, int32_t cols, 
                             Matrix2d& matrix);

    // 创建2维矩阵 初始值为1
    static int8_t CreateOnes(const std::tuple<int32_t, int32_t>& shape, 
                             Matrix2d& matrix);

    // 创建3维矩阵 初始值为1
    static int8_t CreateOnes(int32_t depth, int32_t height, int32_t width,  
                             Matrix3d& matrix);

    // 创建3维矩阵 初始值为1
    static int8_t CreateOnes(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                             Matrix3d& matrix);

    // 打印二维矩阵
    static void MatrixShow(const Matrix2d& matrix);
    
    // 打印三维矩阵
    static void MatrixShow(const Matrix3d& matrix);

    // 打印图像二维矩阵
    static void ImageMatrixShow(const std::vector<std::vector<uint8_t>>& matrix);

    // 2d矩阵相乘  dot product 点积
    static int8_t DotProduct(const Matrix2d& left_matrix, 
                             const Matrix2d& right_matrix, 
                             Matrix2d& result_matrix); 

    // 2d矩阵相乘 函数重载 输入为uint8_t类型 输出为double类型
    static int8_t DotProduct(const std::vector<std::vector<double>>& left_matrix, 
                             const std::vector<std::vector<uint8_t>>& right_matrix, 
                             std::vector<std::vector<double>>& result_matrix); 

    // 2d矩阵对应位置相乘 hadamark积 
    static int8_t HadamarkProduct(const Matrix2d& left_matrix, 
                                  const Matrix2d& right_matrix, 
                                  Matrix2d& result_matrix);

    // 2d矩阵相加 
    static int8_t Add(const Matrix2d& left_matrix, 
                      const Matrix2d& right_matrix, 
                      Matrix2d& result_matrix);

    // 2d矩阵相减
    static int8_t Subtract(const Matrix2d& left_matrix, 
                           const Matrix2d& right_matrix, 
                           Matrix2d& result_matrix);

    // 3d矩阵相减
    static int8_t Subtract(const Matrix3d& left_matrix, 
                           const Matrix3d& right_matrix, 
                           Matrix3d& result_matrix);

    // 矩阵reshape 成几行几列  这是二维矩阵版
    static int8_t Reshape(const Matrix2d& source_matrix,
                          int32_t rows, int32_t cols, 
                          Matrix2d& result_matrix);

    // 矩阵reshape 一维矩阵转二维矩阵版
    static int8_t Reshape(const Matrix1d& source_matrix,
                          int32_t rows, int32_t cols, 
                          Matrix2d& result_matrix);

    // 2维矩阵的装置矩阵
    static int8_t Transpose(const Matrix2d& source_matrix, 
                            Matrix2d& result_matrix);

    // 1个值 乘以 一个2d矩阵
    static int8_t ValueMulMatrix(DataType value,  
                                 const Matrix2d& source_matrix, 
                                 Matrix2d& result_matrix);

    // 1个值 乘以 一个3d矩阵
    static int8_t ValueMulMatrix(DataType value,  
                                 const Matrix3d& source_matrix, 
                                 Matrix3d& result_matrix);

    // 1个值 减去 一个2d矩阵
    static int8_t ValueSubMatrix(DataType value,  
                                 const Matrix2d& source_matrix, 
                                 Matrix2d& result_matrix);

    // 计算2d矩阵的和
    static double Sum(const Matrix2d& source_matrix);

    // 计算均方误差
    static double MeanSquareError(const Matrix2d& output_matrix, 
                                  const Matrix2d& label);
    
    // 补0填充
    static int8_t ZeroPadding(const Matrix3d& source_matrix, 
                              int32_t zero_padding, 
                              Matrix3d& result_matrix);
    
    //得到矩阵中感兴趣的区域 (x, y)起始行列 x+height y+width是结尾行列
    static int8_t GetROI(const Matrix3d& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         Matrix3d& result_matrix);
        
    
    
    
    //判断2d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<double>>& matrix) {
        return false;
    }

    //判断2d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<uint8_t>>& matrix) {
        return true;
    }

    //判断3d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<std::vector<double>>>& matrix) {
        return false;
    }

    //判断3d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<std::vector<uint8_t>>>& matrix) {
        return true;
    }


};    //struct Matrix 





// 检查2d源矩阵或 结果矩阵行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix2d& matrix, 
                                   bool is_write_logging) {
    //先判断 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols is empty";
        return false;
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < rows; i++) {
        if (cols != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查3d源矩阵或 结果矩阵深度 行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix3d& matrix, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channel is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }
    
    int depth = matrix.size();
    int height = matrix[0].size();
    int width = matrix[0][0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < depth; i++) {
        if (height != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < height; j++) {
            if (width != matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查2d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix2d& left_matrix, 
                                   const Matrix2d& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 行 列是否相同
    if (left_matrix.size() != right_matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows is not equal";
        return false;
    }

    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查3d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix3d& left_matrix, 
                                   const Matrix3d& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 深度 行 列是否相同
    if (left_matrix.size() != right_matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < left_matrix[i].size(); j++) {
            if (left_matrix[i][j].size() != right_matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查2d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix2d& matrix, 
                                   int32_t rows, int32_t cols,  
                                   bool is_write_logging) {
    //先判断 行 和 列是否是空
    if (rows <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input rows is empty";
        return false;
    }
    if (cols <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input cols is empty";
        return false;
    }

    //判断矩阵的 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols is empty";
        return false;
    }

    //看看行是否是要求的行
    if (rows != matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix.size(); i++) {
        if (cols != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查3d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix3d& matrix, int32_t depth, 
                                   int32_t height, int32_t width, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (depth <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input depth is empty";
        return false;
    }
    if (height <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input height is empty";
        return false;
    }
    if (width <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input width is empty";
        return false;
    }

    //判断矩阵的 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channel is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }

    //看看行是否是要求的行
    if (depth != matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix.size(); i++) {
        if (height != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < matrix[i].size(); j++) {
            if (width != matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}



//得到二维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t> Matrix<DataType>::GetShape(const Matrix2d& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0);
    }
    
    int height = source_matrix.size();
    int width = source_matrix[0].size();
    
    return std::make_tuple(height, width);
}

//得到三维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t, int32_t> Matrix<DataType>::GetShape(const Matrix3d& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0, 0);
    }
    
    int channel_number = source_matrix.size();
    int height = source_matrix[0].size();
    int width = source_matrix[0][0].size();
    
    return std::make_tuple(channel_number, height, width);
}

//返回一个浮点型的2d矩阵
template <typename DataType>
std::vector<std::vector<double>> Matrix<DataType>::ToDouble(const std::vector<std::vector<uint8_t>>& matrix) {
    //先判断 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG(ERROR) << "matrix check failed, input matrix rows is empty";
        return std::vector<std::vector<double>>(0, std::vector<double>(0));
    }
    if (0 == matrix[0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix cols is empty";
        return std::vector<std::vector<double>>(0, std::vector<double>(0));
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    
    std::vector<std::vector<double>> double_array(rows, std::vector<double>(cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double_array[i][j] = matrix[i][j];           
            }
        }
    }

    return double_array;
}

//返回一个浮点型的3d矩阵
template <typename DataType>
std::vector<std::vector<std::vector<double>>> Matrix<DataType>::ToDouble(const std::vector<std::vector<std::vector<uint8_t>>>& matrix) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG(ERROR) << "matrix check failed, input matrix channel is empty";
        return std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0)));
    }
    if (0 == matrix[0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix height is empty";
        return std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0)));
    }
    if (0 == matrix[0][0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix width is empty";
        return std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0)));
    }

    int depth = matrix.size();
    int height = matrix[0].size();
    int width = matrix[0][0].size();
    
    std::vector<std::vector<std::vector<double>>> double_array(depth, std::vector<std::vector<double>>(height), 
                                                               std::vector<double>(width));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    double_array[i][j][k] = matrix[i][j][k];           
                }
            }
        }
    }

    return double_array;
}

// 创建2d矩阵 初始值0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(int32_t rows, int32_t cols, 
                                     Matrix2d& matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 0));

    return 0;
}

// 创建2d矩阵 初始值0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(const std::tuple<int32_t, int32_t>& shape, 
                                     Matrix2d& matrix) {
    int rows = -1;
    int cols = -1;
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 0));

    return 0;
}

// 创建3维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(int32_t depth, int32_t height, int32_t width, 
                                     Matrix3d& matrix) {
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 0)));

    return 0;
}

// 创建3维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                                     Matrix3d& matrix) {
    int depth = -1;
    int height = -1;
    int width = -1;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 0)));

    return 0;
}

//创建2维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(int32_t rows, int32_t cols, 
                                    Matrix2d& matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 1));

    return 0;
}

//创建2维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(const std::tuple<int32_t, int32_t>& shape, 
                                    Matrix2d& matrix) {
    int rows = -1;
    int cols = -1;
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 1));

    return 0;
}

// 创建3维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(int32_t depth, int32_t height, int32_t width, 
                                    Matrix3d& matrix) {
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 1)));

    return 0;
}

// 创建3维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                                    Matrix3d& matrix) {
    int depth = -1;
    int height = -1;
    int width = -1;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 1)));

    return 0;
}

//打印二维矩阵
template <typename DataType>
void Matrix<DataType>::MatrixShow(const Matrix2d& matrix) {
    //check 源矩阵
    if (!MatrixCheck(matrix, true)) {
        LOG(ERROR) << "print matrix failed";
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

//打印三维矩阵
template <typename DataType>
void Matrix<DataType>::MatrixShow(const Matrix3d& matrix) {
    //getshape 会check
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "print matrix failed";
        return ;
    }
    //元祖解包
    int channel_number;
    int height;
    int width;
    std::tie(channel_number, height, width) = shape;

    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                if (0 == i
                        && 0 == j
                        && 0 == k) {
                    std::cout << "[[[";
                } else if (0 == j
                            && 0 == k) {
                    std::cout << " [[";
                } else if (0 == k) {
                    std::cout << "  [";
                }

                //如果是负数 则会多占一格 那么是正数 就多加个空格
                //设置浮点的格式 后面6位
                int space_number = 0;
                if (matrix[i][j][k] >= 0) {
                    if ((matrix[i][j][k] / 10.0) < 1.0) {
                        space_number += 3;
                    } else {
                        space_number += 2;
                    }
                } else {
                    if ((matrix[i][j][k] / 10.0) < 1.0) {
                        space_number += 2;
                    } else {
                        space_number += 1;
                    }
                }

                std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                          << std::setprecision(8) << std::string(space_number, ' ')
                          << matrix[i][j][k];
           
                if ((i + 1) == channel_number 
                        && (j + 1) == height
                        && (k + 1) == width) {
                    std::cout << "  ]]]" << std::endl;
                } else if ((j + 1) == height 
                            && (k + 1) == width) {
                    std::cout << "  ]]\n" << std::endl;
                } else if ((k + 1) == width) {
                    std::cout << "  ]" << std::endl;
                }
            }
        }
    }
    std::cout << std::endl << std::endl;
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
                      << static_cast<int>(matrix[i][j]) << " ";
            
            if ((j + 1) == matrix[i].size()) {
                std::cout << " ]" << std::endl;
            }
        }
    }
    std::cout << std::endl;
}


//矩阵相乘(dot product)点积
template <class DataType>
int8_t Matrix<DataType>::DotProduct(const Matrix2d& left_matrix, 
                                    const Matrix2d& right_matrix, 
                                    Matrix2d& result_matrix) {
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(left_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    auto right_matrix_shape = GetShape(right_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int i = 0; i < left_matrix_rows; i++) {
            for (int j = 0; j < right_matrix_cols; j++) {
                for (int k = 0; k < left_matrix_cols; k++) {
                    result_matrix[i][j] += left_matrix[i][k] * right_matrix[k][j];
                }
            }
        }
    }

    return 0;
}

//矩阵相乘函数重载
template <class DataType>
int8_t Matrix<DataType>::DotProduct(const std::vector<std::vector<double>>& left_matrix, 
                                    const std::vector<std::vector<uint8_t>>& right_matrix, 
                                    std::vector<std::vector<double>>& result_matrix) {
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(left_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    auto right_matrix_shape = GetShape(ToDouble(right_matrix));
    if (right_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int i = 0; i < left_matrix_rows; i++) {
            for (int j = 0; j < right_matrix_cols; j++) {
                for (int k = 0; k < left_matrix_cols; k++) {
                    result_matrix[i][j] += left_matrix[i][k] * right_matrix[k][j];
                }
            }
        }
    }

    return 0;
}

//hadamark积 也就是矩阵相应位置相乘
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const Matrix2d& left_matrix, 
                                         const Matrix2d& right_matrix, 
                                         Matrix2d& result_matrix) {
    //检查 两个输入矩阵的正确性 内部会先check单独矩阵的行 列 
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix hadamark product failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                result_matrix[i][j] = left_matrix[i][j] * right_matrix[i][j];
            }
        }
    }

    return 0;
}

//矩阵相加
template <typename DataType>
int8_t Matrix<DataType>::Add(const Matrix2d& left_matrix, 
                             const Matrix2d& right_matrix, 
                             Matrix2d& result_matrix) {
    //检查 两个输入矩阵的正确性
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix add failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵相加
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                result_matrix[i][j] = left_matrix[i][j] + right_matrix[i][j];
            }
        }
    }

    return 0;
}

//矩阵相减
template <typename DataType>
int8_t Matrix<DataType>::Subtract(const Matrix2d& left_matrix, 
                                  const Matrix2d& right_matrix, 
                                  Matrix2d& result_matrix) {
    //检查 两个输入矩阵的正确性
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix subtract failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }
        
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵相减
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                result_matrix[i][j] = left_matrix[i][j] - right_matrix[i][j];
            }
        }
    }

    return 0;
}

//矩阵相减
template <typename DataType>
int8_t Matrix<DataType>::Subtract(const Matrix3d& left_matrix, 
                                  const Matrix3d& right_matrix, 
                                  Matrix3d& result_matrix) {
    //检查 两个输入矩阵的正确性
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix subtract failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }
        
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵相减
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                for (int k = 0; k < left_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = left_matrix[i][j][k] - right_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}

//matrix reshape  把原矩阵变成 几行几列
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix2d& source_matrix,
                                 int32_t rows, int32_t cols, 
                                 Matrix2d& result_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }
    
    //check一下源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "reshape matrix failed";
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
    
    //check一下输出数组
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    }
    
    index = 0;
    //reshape 把一维数组值赋给新数组  
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = matrix_data[index++];    
        }
    }

    return 0;
}

//接收输入矩阵为1维矩阵 函数重载
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix1d& source_matrix,
                                 int32_t rows, int32_t cols, 
                                 Matrix2d& result_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    //check 源矩阵
    if (0 == source_matrix.size()) {
        LOG(ERROR) << "reshape matrix failed, input matrix is empty";
        return -1;
    }

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = source_matrix.size();
    
    if (matrix_total_size != (rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }
    
    result_matrix.clear();
    result_matrix = Matrix2d(rows, Matrix1d(cols));

    int index = 0;
    //再赋值给新数组
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = source_matrix[index++];    
        }
    }

    return 0;
}

//转置矩阵
template <typename DataType>
int8_t Matrix<DataType>::Transpose(const Matrix2d& source_matrix, 
                                   Matrix2d& result_matrix) {
    int source_matrix_rows = 0;
    int source_matrix_cols = 0;
    //reshape 会check
    auto shape = GetShape(source_matrix);
    if (shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "transpose matrix failed";
        return -1;
    }
    //元组解包
    std::tie(source_matrix_rows, source_matrix_cols) = shape;

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

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //转置矩阵
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                result_matrix[j][i] = source_matrix[i][j];
            }
        }
    }

    return 0;
}

//2d矩阵都乘以一个值
template <typename DataType>
int8_t Matrix<DataType>::ValueMulMatrix(DataType value,  
                                        const Matrix2d& source_matrix, 
                                        Matrix2d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value mul matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                result_matrix[i][j] = source_matrix[i][j] * value;
            }
        }
    }

    return 0;
}

//3d矩阵都乘以一个值
template <typename DataType>
int8_t Matrix<DataType>::ValueMulMatrix(DataType value,  
                                        const Matrix3d& source_matrix, 
                                        Matrix3d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value mul matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    } 

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                for (int k = 0; k < source_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = source_matrix[i][j][k] * value;
                }
            }
        }
    }

    return 0;
}

//一个值减去矩阵每个值
template <typename DataType>
int8_t Matrix<DataType>::ValueSubMatrix(DataType value,  
                                        const Matrix2d& source_matrix, 
                                        Matrix2d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value sub matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    } 

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                result_matrix[i][j] = value - source_matrix[i][j];
            }
        }
    }

    return 0;
}

//计算2d矩阵的和
template <typename DataType>
double Matrix<DataType>::Sum(const Matrix2d& source_matrix) {
    double sum = 0.0;

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum) 
        //多线程数据竞争  加锁保护
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                sum += source_matrix[i][j];
            }
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
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum)
        //多线程数据竞争  加锁保护
        for (int i = 0; i < output_matrix.size(); i++) {
            for (int j = 0; j < output_matrix[i].size(); j++) {
                 sum += pow((label[i][j] - output_matrix[i][j]), 2);
            }
        }
    }

    return sum / 2;
}

// 补0填充
template <typename DataType>
int8_t Matrix<DataType>::ZeroPadding(const Matrix3d& source_matrix, 
                                     int32_t zero_padding, 
                                     Matrix3d& result_matrix) {
    //如果外圈补0是0的话 就不用补
    if (0 == zero_padding) {
        result_matrix = source_matrix;
        return 0;
    }
    //getshape 会check一下源矩阵
    auto shape = GetShape(source_matrix);
    int depth;
    int height;
    int width;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix depth is empty";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix height is empty";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix width is empty";
        return -1;
    }

    //check一下结果矩阵 深度应该不变 行 列 应该加上zero_padding
    if (!MatrixCheck(result_matrix, depth, height + 2 * zero_padding, width + 2 * zero_padding)) {
        result_matrix.clear();
        result_matrix = Matrix3d(depth, Matrix2d(height + 2 * zero_padding, Matrix1d(width + 2 * zero_padding, 0)));
    }

     
}

//得到矩阵中感兴趣的区域 (x, y)起始行列 x+height y+width是结尾行列
template <typename DataType>
int8_t Matrix<DataType>::GetROI(const Matrix3d& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                Matrix3d& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "get matrix roi failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "get matrix roi failed, input y < 0";
        return -1;
    }
    if (height < x) {
        LOG(ERROR) << "get matrix roi failed, input height < x";
        return -1;
    }
    if (width < y) {
        LOG(ERROR) << "get matrix roi failed, input width < y";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_depth;
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_depth, source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_depth <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix depth is empty";
        return -1;
    }
    if (source_matrix_height <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix height is empty";
        return -1;
    }
    if (source_matrix_width <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix width is empty";
        return -1;
    }

    //这里要多判断一个 x + height 不能超过了矩阵的行 y + width 不能超过了矩阵的列
    if (x + height > source_matrix_height) {
        LOG(ERROR) << "get matrix roi failed, input matrix x + height > source matrix height";
        return -1;
    }
    if (y + width > source_matrix_width) {
        LOG(ERROR) << "get matrix roi failed, input matrix y + width > source matrix width";
        return -1;
    }
    
    //check一下结果矩阵
    if (!MatrixCheck(result_matrix, source_matrix_depth, height, width, false)) {
        result_matrix.clear();
        result_matrix = Matrix3d(source_matrix_depth, Matrix2d(height, Matrix1d(width)));
    }
 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        for (int i = 0; i < source_matrix_depth; i++) {
            for (int j = x, q = 0; j < x + height; j++, q++) {
                for (int k = y, w = 0; k < y + width; k++, w++) {
                    result_matrix[i][q][w] = source_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
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
    static int8_t Normal(float mean, float stddev, int32_t rows, int32_t cols, 
                         Matrix2d& random_matrix);

    //生成随机的浮点数二维矩阵
    static int8_t Uniform(float a, float b, int32_t rows, int32_t cols, 
                          Matrix2d& random_matrix);

    //生成随机的浮点数三维矩阵
    static int8_t Uniform(float a, float b, int32_t channel_number, 
                          int32_t height, int32_t width, 
                          Matrix3d& random_matrix);

    //生成随机的整数二维矩阵
    static int8_t RandInt(float a, float b, int32_t rows, int32_t cols, 
                          Matrix2d& random_matrix);


};   //struct Random

//生成服从正态分布的随机数二维矩阵
template <typename DataType>
int8_t Random<DataType>::Normal(float mean, float stddev, int32_t rows, int32_t cols, 
                                Matrix2d& random_matrix) {
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


//生成一个 rows * cols的随机数二维矩阵 值的范围在a 到 b之间 
template <typename DataType>
int8_t Random<DataType>::Uniform(float a, float b, int32_t rows, int32_t cols, 
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

//生成一个 rows * cols的随机数三维矩阵 值的范围在a 到 b之间 
template <typename DataType>
int8_t Random<DataType>::Uniform(float a, float b, int32_t channel_number, 
                                 int32_t height, int32_t width, 
                                 Matrix3d& random_matrix) {
                
    if (channel_number <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input width <= 0";
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
    random_matrix = Matrix3d(channel_number, Matrix2d(height, Matrix1d(width)));
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_real_distribution<double> generate_random(a, b);
    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                random_matrix[i][j][k] = generate_random(random_engine);
            }
        }
    }

    return 0;
}

//生成一个随机整数二维矩阵
template <typename DataType>
int8_t Random<DataType>::RandInt(float a, float b, int32_t rows, int32_t cols,  
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

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //计算 1 / (1 + exp(-input_array))
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                //exp返回e的x次方 得到0. 1. 2.值 加上1都大于1了 然后用1除  最后都小于1
                output_array[i][j] = 1.0 / (1.0 + exp(-input_array[i][j])); 
            }
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

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //计算 output(1 - output)
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                delta_array[i][j] = output_array[i][j] * (1.0 - output_array[i][j]);
            }
        }
    }
}

//sigmoid激活函数的反向计算
template <typename DataType>
void Activator<DataType>::SigmoidBackward(const ImageMatrix2d& output_array, 
                                          Matrix2d& delta_array) {
    //如果输出数组未初始化 
    if (0 == delta_array.size()) {
        auto shape = matrix::Matrix<uint8_t>::GetShape(output_array);
        matrix::Matrix<double>::CreateZeros(shape, delta_array);
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //计算 output(1 - output)
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                delta_array[i][j] = output_array[i][j] * (1.0 - output_array[i][j]);
            }
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
           (int)ptime->tm_mday,        (int)ptime->tm_hour, 
           (int)ptime->tm_min,         (int)ptime->tm_sec);
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
