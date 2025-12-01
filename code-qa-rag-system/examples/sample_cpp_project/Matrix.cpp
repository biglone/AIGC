/**
 * Matrix.cpp
 *
 * 矩阵类的实现
 */

#include "Matrix.h"
#include <iomanip>

// 构造函数：创建 rows x cols 的矩阵，初始化为0
Matrix::Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols) {
    // 初始化为零矩阵
    data.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        data[i].resize(cols, 0.0);
    }
}

// 构造函数：从二维vector创建
Matrix::Matrix(const std::vector<std::vector<double>>& data)
    : data(data) {
    if (data.empty() || data[0].empty()) {
        throw std::invalid_argument("矩阵不能为空");
    }

    rows = data.size();
    cols = data[0].size();

    // 检查所有行的列数是否一致
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("矩阵的所有行必须有相同的列数");
        }
    }
}

// 访问元素（可修改）
double& Matrix::operator()(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("矩阵索引越界");
    }
    return data[i][j];
}

// 访问元素（只读）
const double& Matrix::operator()(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("矩阵索引越界");
    }
    return data[i][j];
}

// 矩阵加法
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("矩阵加法要求两个矩阵形状相同");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

// 矩阵减法
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("矩阵减法要求两个矩阵形状相同");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

// 矩阵乘法
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument(
            "矩阵乘法要求：第一个矩阵的列数 = 第二个矩阵的行数"
        );
    }

    Matrix result(rows, other.cols);

    // 标准矩阵乘法：O(n^3)
    // result[i][j] = sum(this[i][k] * other[k][j])
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

// 标量乘法
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

// 矩阵转置
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);  // 注意：行列互换
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];  // 转置：行列互换
        }
    }
    return result;
}

// 打印矩阵
void Matrix::print() const {
    std::cout << "Matrix (" << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::setprecision(2)
                      << std::fixed << data[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// 创建单位矩阵
Matrix Matrix::identity(size_t n) {
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = 1.0;  // 对角线为1
    }
    return result;
}

// 创建零矩阵
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);  // 构造函数已经初始化为0
}

// 创建全1矩阵
Matrix Matrix::ones(size_t rows, size_t cols) {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = 1.0;
        }
    }
    return result;
}
