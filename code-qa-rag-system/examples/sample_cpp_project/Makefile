# Makefile for Matrix Library Example

# 编译器
CXX = g++

# 编译选项
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# 目标可执行文件
TARGET = matrix_demo

# 源文件
SOURCES = main.cpp Matrix.cpp

# 对象文件
OBJECTS = $(SOURCES:.cpp=.o)

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)
	@echo "✅ 编译完成: $(TARGET)"

# 编译
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 运行
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(OBJECTS) $(TARGET)
	@echo "✅ 清理完成"

# 重新编译
rebuild: clean all

.PHONY: all run clean rebuild
