/*
 * =====================================================================================
 *
 *       Filename:  normalizer.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年12月30日 08时59分46秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef UTILITY_NORMALIZER_HPP_
#define UTILITY_NORMALIZER_HPP_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "matrix_math_function.hpp"

namespace utility {
//归一化
class Normalizer {
public:
    //使用默认构造 默认移动构造和移动赋值  删除拷贝构造 和拷贝赋值
    Normalizer() {}
    ~Normalizer() {}
    Normalizer(const Normalizer&) = delete;
    Normalizer& operator =(const Normalizer&) = delete;
    Normalizer(Normalizer&&) = default;
    Normalizer& operator =(Normalizer&&) = default;

public:
    //位运算 输入为0-255的值 8位二进制从低位到高位 位是1就是0.9 位是0就是0.1 输出一个8维列表
    static void Normalize(uint8_t number, std::vector<std::vector<double>>&);

    //这个是对外提供的接口 
    static void Normalize(uint8_t number, size_t rows, size_t cols, 
                          std::vector<std::vector<double>>&);

    //输入为模型预测的输出值 二值化然后把值还原成最初的number 如果相等 则表明每位的预测偏差没有很大
    static uint8_t Denormalize(const std::vector<std::vector<double>>& model_predict_output);

private:
    static std::vector<uint8_t> mask_;
};

//类外定义静态成员  因为静态成员是属于类的成员只有一份 而不是实例化对象的成员
std::vector<uint8_t> Normalizer::mask_{0x1, 0x2, 0x4, 0x8, 
                                       0x10, 0x20, 0x40, 0x80};

void Normalizer::Normalize(uint8_t number, 
                           std::vector<std::vector<double>>& result_matrix) {
    std::vector<double> data;
    data.reserve(8);
    for (const auto mask : mask_) {
        //如果相与值大于0 就是0.9 代表这个值这位是1 否则是0.1 代表这个值这位是0
        if (number & mask) {
            data.push_back(0.9);
        } else {
            data.push_back(0.1);
        }
    }

    Matrix::MatrixReshape(data, 8, 1, result_matrix);
}

void Normalizer::Normalize(uint8_t number, size_t rows, size_t cols, 
                           std::vector<std::vector<double>>& label) {
    std::vector<double> data;
    data.reserve(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        //如果值等于结果值 那个地方就是0.9 否则就是0.1
        if (number == i) {
            data.push_back(0.9);
        } else {
            data.push_back(0.1);
        }
    }
    
    Matrix::MatrixReshape(data, rows, cols, label);
}

uint8_t Normalizer::Denormalize(const std::vector<std::vector<double>>& model_predict_output) {
    //二值化  此时传入的矩阵是模型预测的输出值 
    //通过大于0.5取1 小于0.5取0的方式 然后把值加起来还原最开始的值 来看是否改变了
    std::vector<uint8_t> binary_array;
    binary_array.reserve(8);
    for (const auto& output : model_predict_output) {
        for (auto i : output) {
            if (i > 0.5) {
                binary_array.push_back(1);
            } else {
                binary_array.push_back(0);
            }
        }
    }
    
    uint8_t number = 0;
    for (int i = 0; i < binary_array.size(); i++) {
        number += binary_array[i] * mask_[i];
    }

    return number;
}

}        //namespace utility

#endif   //UTILITY_NORMALIZER_HPP_
