/**
 * Matrix.h
 *
 * 简单的矩阵类实现
 * 支持基本的矩阵运算：加法、乘法、转置等
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <iostream>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    /**
     * 构造函数：创建 rows x cols 的矩阵，初始化为0
     */
    Matrix(size_t rows, size_t cols);

    /**
     * 构造函数：从二维vector创建矩阵
     */
    Matrix(const std::vector<std::vector<double>>& data);

    /**
     * 获取矩阵的行数
     */
    size_t getRows() const { return rows; }

    /**
     * 获取矩阵的列数
     */
    size_t getCols() const { return cols; }

    /**
     * 访问元素（可修改）
     * 用法：matrix(i, j) = value;
     */
    double& operator()(size_t i, size_t j);

    /**
     * 访问元素（只读）
     * 用法：double val = matrix(i, j);
     */
    const double& operator()(size_t i, size_t j) const;

    /**
     * 矩阵加法
     * 返回：this + other
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * 矩阵减法
     * 返回：this - other
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * 矩阵乘法
     * 返回：this * other
     * 要求：this的列数 = other的行数
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * 标量乘法
     * 返回：this * scalar
     */
    Matrix operator*(double scalar) const;

    /**
     * 矩阵转置
     * 返回：转置后的矩阵
     */
    Matrix transpose() const;

    /**
     * 打印矩阵
     */
    void print() const;

    /**
     * 创建单位矩阵
     * 参数：n - 矩阵大小 (n x n)
     */
    static Matrix identity(size_t n);

    /**
     * 创建零矩阵
     */
    static Matrix zeros(size_t rows, size_t cols);

    /**
     * 创建全1矩阵
     */
    static Matrix ones(size_t rows, size_t cols);
};

#endif // MATRIX_H
