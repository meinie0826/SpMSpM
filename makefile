CXX = nvcc
CXXFLAGS = -std=c++14 -O3 -arch=sm_80
INCLUDES = -I./csrc -I./kernel_benchmark

# 默认目标
all: spmm_test

# 源文件
SRCS = csrc/SpMM_API.cu csrc/SpMSpM_API.cu csrc/spmm_test.cu

# 编译测试程序
spmm_test: $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRCS)

# 编译示例程序
spmm_example: spmm_example.cu
	$(CXX) $(CXXFLAGS) -o $@ $< -I.

# 清理目标文件和可执行文件
clean:
	rm -f spmm_example spmm_test

.PHONY: all clean