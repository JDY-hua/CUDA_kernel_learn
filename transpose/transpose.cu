#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cassert>

/* 
定义：遵从onnx算子定义https://onnx.ai/onnx/operators/onnx__Transpose.html
参数：
input ： 指针地址+输入的属性（如shape）
output:  指针地址+输出的属性（如shape）
attrib： 固有参数（本算子为permute参数)
实现思路：
实现一个transpose的op，然后调用transpose的kernel
*/

// naive
/*
paramters:
in_shape:[batchsize,head_nums,seqlen,size_per_head]
out_shape:[batchsize, head_nums, size_per_head, seqlen]
permute:[0,1,3,2]
*/

using namespace std;

template <typename T>
bool transpose_naive_op(
    T* in_data, T* out_data,
    vector<int>& in_shape, vector<int>& out_shape,
    vector<int>& permute);


template <int NUM_DIMS, typename T>
__global__ void transpose_naive_kernel(T* in_data, T* out_data, 
    int* strides_in, int* strides_out,
    int* permute, int nums);

// 测试用例
void test_transpose_naive_op() {
    // 定义输入和输出的shape
    vector<int> in_shape = {128, 32, 256, 256};
    vector<int> out_shape = {128, 32, 256, 256};
    vector<int> permute = {0, 1, 3, 2};

    // 计算输入数据的总元素个数
    int num_elements = 1;
    for (int dim : in_shape) {
        num_elements *= dim;
    }

    // 创建输入数据
    thrust::host_vector<float> h_in_data(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        h_in_data[i] = static_cast<float>(i);
    }

    // 创建输出数据
    thrust::host_vector<float> h_out_data(num_elements);

    // 将输入数据复制到设备
    thrust::device_vector<float> d_in_data = h_in_data;
    thrust::device_vector<float> d_out_data = h_out_data;

    // 调用transpose_naive_op函数
    bool success = transpose_naive_op(
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_out_data.data()),
        in_shape, out_shape, permute);

    // 将输出数据复制回主机
    thrust::copy(d_out_data.begin(), d_out_data.end(), h_out_data.begin());

    // 验证输出结果
    if (success) {
        // 计算期望的输出数据
        thrust::host_vector<float> expected_out_data(num_elements);
        int idx = 0;
        for (int b = 0; b < in_shape[0]; ++b) {
            for (int h = 0; h < in_shape[1]; ++h) {
                for (int s = 0; s < in_shape[3]; ++s) {
                    for (int p = 0; p < in_shape[2]; ++p) {
                        int in_idx = b * in_shape[1] * in_shape[2] * in_shape[3] +
                                     h * in_shape[2] * in_shape[3] +
                                     p * in_shape[3] + s;
                        expected_out_data[idx++] = h_in_data[in_idx];
                    }
                }
            }
        }

        // 比较实际输出和期望输出
        for (int i = 0; i < num_elements; ++i) {
            assert(h_out_data[i] == expected_out_data[i]);
        }

        std::cout << "Test passed!" << std::endl;
    } else {
        std::cerr << "Test failed!" << std::endl;
    }
}

int main() {
    test_transpose_naive_op();
    return 0;
}

template <typename T>
bool transpose_naive_op(
    T* in_data, T* out_data,
    vector<int>& in_shape, vector<int>& out_shape,
    vector<int>& permute) {
    if (in_shape.size() != out_shape.size()) {
        return false;
    }
    const int dims = 4;

    auto shape2stride = [&](vector<int>& shape, int dims) -> vector<int> {
        vector<int> strides(dims, 1);
        for (int i = dims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    thrust::host_vector<int> h_in_strides = shape2stride(in_shape, dims);
    thrust::host_vector<int> h_out_strides = shape2stride(out_shape, dims);
    thrust::host_vector<int> h_permute = permute;

    auto com_nums = [](vector<int>& shape) -> const int {
        int nums = 1;
        for (int i = 0; i < shape.size(); ++i) {
            nums = nums * shape[i];
        }
        return nums;
    };
    const int nums = com_nums(in_shape);

    thrust::device_vector<int> d_in_strides = h_in_strides;
    thrust::device_vector<int> d_out_strides = h_out_strides;
    thrust::device_vector<int> d_permute = h_permute;

    int BLOCK_SIZE = 256;
    int grid_size = (nums + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 调用内核函数
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    transpose_naive_kernel<dims, T><<<grid_size, BLOCK_SIZE>>>(
        in_data, out_data,
        thrust::raw_pointer_cast(d_in_strides.data()),
        thrust::raw_pointer_cast(d_out_strides.data()),
        thrust::raw_pointer_cast(d_permute.data()),
        nums);

    // 同步设备并检查错误
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

template <int NUM_DIMS, typename T>
__global__ void transpose_naive_kernel(T* in_data, T* out_data, 
    int* strides_in, int* strides_out,
    int* permute, int nums) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nums) {
        int offset_out = tid;
        int offset_tmp = offset_out;
        int offset_in = 0;
        for (int i = 0; i < NUM_DIMS; ++i) {
            offset_in += (offset_tmp / strides_out[i]) * strides_in[permute[i]];
            offset_tmp %= strides_out[i];
        }
        out_data[offset_out] = in_data[offset_in];
    }
}


// template <int NUM_DIMS, typename T>
// __gloabl__ void 