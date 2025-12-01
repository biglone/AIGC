/**
 * main.cpp
 *
 * 矩阵类的使用示例
 */

#include "Matrix.h"
#include <iostream>

int main() {
    std::cout << "========================================\n";
    std::cout << "矩阵类使用示例\n";
    std::cout << "========================================\n\n";

    try {
        // 示例1：创建矩阵
        std::cout << "[示例1] 创建矩阵\n";
        std::cout << "-------------------\n";

        Matrix A(2, 3);  // 2x3 零矩阵
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(1, 0) = 4.0;
        A(1, 1) = 5.0;
        A(1, 2) = 6.0;

        std::cout << "矩阵 A:\n";
        A.print();

        // 示例2：从vector创建
        std::cout << "[示例2] 从vector创建\n";
        std::cout << "-------------------\n";

        std::vector<std::vector<double>> data = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        };
        Matrix B(data);

        std::cout << "矩阵 B:\n";
        B.print();

        // 示例3：矩阵乘法
        std::cout << "[示例3] 矩阵乘法\n";
        std::cout << "-------------------\n";

        Matrix C = A * B;  // A(2x3) * B(3x2) = C(2x2)

        std::cout << "C = A * B:\n";
        C.print();

        // 示例4：矩阵转置
        std::cout << "[示例4] 矩阵转置\n";
        std::cout << "-------------------\n";

        Matrix A_T = A.transpose();

        std::cout << "A 的转置:\n";
        A_T.print();

        // 示例5：矩阵加法
        std::cout << "[示例5] 矩阵加法\n";
        std::cout << "-------------------\n";

        Matrix D = C + C;

        std::cout << "D = C + C:\n";
        D.print();

        // 示例6：标量乘法
        std::cout << "[示例6] 标量乘法\n";
        std::cout << "-------------------\n";

        Matrix E = C * 2.0;

        std::cout << "E = C * 2.0:\n";
        E.print();

        // 示例7：单位矩阵
        std::cout << "[示例7] 单位矩阵\n";
        std::cout << "-------------------\n";

        Matrix I = Matrix::identity(3);

        std::cout << "3x3 单位矩阵:\n";
        I.print();

        // 示例8：验证矩阵乘法性质：A * I = A
        std::cout << "[示例8] 验证 A * I = A\n";
        std::cout << "-------------------\n";

        Matrix I2 = Matrix::identity(3);
        Matrix A_copy = A * I2;

        std::cout << "A * I:\n";
        A_copy.print();

        std::cout << "原始 A:\n";
        A.print();

        std::cout << "✅ 所有示例执行成功！\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
