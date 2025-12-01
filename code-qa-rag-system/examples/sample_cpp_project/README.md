# 示例C++项目 - 简单矩阵库

这是一个简单的C++矩阵库，用于演示代码库问答系统的功能。

## 📁 文件结构

```
sample_cpp_project/
├── Matrix.h       # 矩阵类声明
├── Matrix.cpp     # 矩阵类实现
├── main.cpp       # 使用示例
├── Makefile       # 编译脚本
└── README.md      # 本文件
```

## 🚀 快速开始

### 编译

```bash
make
```

### 运行

```bash
./matrix_demo
```

### 清理

```bash
make clean
```

## 📚 功能说明

### Matrix类

**构造函数**：
- `Matrix(size_t rows, size_t cols)` - 创建零矩阵
- `Matrix(const std::vector<std::vector<double>>& data)` - 从二维数组创建

**基本操作**：
- `operator()` - 访问/修改元素
- `getRows()` / `getCols()` - 获取维度
- `print()` - 打印矩阵

**矩阵运算**：
- `operator+` - 矩阵加法
- `operator-` - 矩阵减法
- `operator*` - 矩阵乘法
- `operator*` - 标量乘法
- `transpose()` - 转置

**静态方法**：
- `identity(n)` - 创建n×n单位矩阵
- `zeros(rows, cols)` - 创建零矩阵
- `ones(rows, cols)` - 创建全1矩阵

## 💡 使用示例

### 创建矩阵

```cpp
// 方法1：创建零矩阵，然后赋值
Matrix A(2, 3);
A(0, 0) = 1.0;
A(0, 1) = 2.0;
// ...

// 方法2：从vector创建
std::vector<std::vector<double>> data = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0}
};
Matrix B(data);
```

### 矩阵运算

```cpp
// 加法
Matrix C = A + B;

// 乘法（要求：A的列数 = B的行数）
Matrix D = A * B;

// 转置
Matrix A_T = A.transpose();

// 标量乘法
Matrix E = A * 2.0;
```

### 特殊矩阵

```cpp
// 3x3 单位矩阵
Matrix I = Matrix::identity(3);

// 2x3 零矩阵
Matrix Z = Matrix::zeros(2, 3);

// 2x3 全1矩阵
Matrix O = Matrix::ones(2, 3);
```

## 🔍 测试代码问答系统

使用这个项目测试代码问答系统：

### 1. 索引代码库

```bash
cd project_code_qa
python code_indexer.py
```

选择索引 `examples/sample_cpp_project` 目录。

### 2. 提问示例

**理解代码**：
- "Matrix类有哪些主要功能？"
- "如何创建一个矩阵？"
- "矩阵乘法是如何实现的？"

**查找用法**：
- "transpose()函数怎么用？"
- "如何创建单位矩阵？"
- "矩阵加法需要什么条件？"

**代码审查**：
- "Matrix类的实现有什么可以优化的？"
- "代码中有什么潜在的性能问题？"
- "异常处理是否完善？"

**技术细节**：
- "矩阵乘法的时间复杂度是多少？"
- "为什么有两个operator()重载？"
- "transpose()函数的原理是什么？"

## 📝 已知问题

这是一个教学示例，存在以下限制：

1. **性能**：使用简单的O(n³)矩阵乘法，未优化
2. **功能**：缺少行列式、逆矩阵等高级功能
3. **内存**：使用std::vector，内存布局不连续

## 🎯 可能的优化方向

1. **性能优化**：
   - 使用Strassen算法优化矩阵乘法
   - 使用SIMD指令（AVX/SSE）
   - 实现分块矩阵乘法

2. **功能扩展**：
   - 添加行列式计算
   - 添加矩阵求逆
   - 添加LU分解

3. **内存优化**：
   - 使用连续内存存储（一维数组）
   - 实现移动语义
   - 添加内存池

## 📚 相关资源

- [矩阵运算基础](https://en.wikipedia.org/wiki/Matrix_(mathematics))
- [C++运算符重载](https://en.cppreference.com/w/cpp/language/operators)
- [RAII和异常安全](https://en.cppreference.com/w/cpp/language/raii)

---

**这个项目是代码库问答系统的测试样例**
