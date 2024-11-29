## nvidia-nsight
### PTX和SASS分别是什么？

#### PTX（Parallel Thread Execution）####

**定义**：
- PTX（Parallel Thread Execution）是一种中间表示（IR），用于描述GPU的并行计算指令。它是一种类似于汇编语言的低级编程语言，但比实际的机器代码更抽象。

**特点**：
- **平台无关性**：PTX代码是平台无关的，可以在不同的NVIDIA GPU架构上运行。编译器可以将PTX代码编译成特定GPU架构的机器代码。
- **可移植性**：由于PTX是中间表示，开发者可以编写一次PTX代码，然后在不同的GPU架构上运行，而不需要为每个架构编写不同的代码。
- **优化**：PTX允许编译器在编译时进行高级优化，以生成更高效的机器代码。

**用途**：
- PTX通常用于高级编程语言（如CUDA）编译后的中间步骤。开发者通常不会直接编写PTX代码，而是通过CUDA C/C++等高级语言编写代码，然后由编译器生成PTX。

#### SASS（Shader Assembly）####

**定义**：
- SASS（Shader Assembly）是NVIDIA GPU的实际机器代码，也称为“机器码”或“二进制代码”。它是GPU硬件可以直接执行的指令集。

**特点**：
- **平台相关性**：SASS代码是特定于GPU架构的，不同的GPU架构（如Kepler、Maxwell、Pascal等）有不同的SASS指令集。
- **低级**：SASS是GPU的实际执行代码，开发者通常不会直接编写SASS代码，而是通过高级语言或PTX生成。
- **性能优化**：由于SASS是直接在硬件上执行的代码，因此它对性能有直接影响。编译器会尽可能优化SASS代码以提高性能。

**用途**：
- SASS通常用于调试和性能分析。开发者可以通过工具（如NVIDIA Nsight）查看和分析生成的SASS代码，以了解程序在GPU上的实际执行情况。

#### 总结

- **PTX**：是一种中间表示，用于描述GPU的并行计算指令，具有平台无关性和可移植性。
- **SASS**：是NVIDIA GPU的实际机器代码，具有平台相关性，直接在硬件上执行。

开发者通常通过高级编程语言（如CUDA）编写代码，然后由编译器生成PTX，最终编译成SASS以在GPU上执行。

### CUDA中的l1 cache和l2 cache是什么？如何借助其优化算子


### relu算子中的relu_f16x8_pack_kernel的原理


### 结果中为什么f16x8比f16x2的时间长，f16x2为什么比f16x8_pack的时间长
问题一：初步认为是线程数量的问题。
问题二：
### nvcc、nsys、ncu的参数都代表啥
#### nvcc
--generate-line-info 是 NVIDIA CUDA 编译器（nvcc）的一个选项，用于在生成的设备代码（PTX 或 SASS）中包含行号信息。这些行号信息可以帮助调试工具（如 cuda-gdb 或 Nsight Debugger）在调试过程中将设备代码与源代码进行关联，从而更方便地进行调试
#### nsys
nsys profile
nsys profile：这是 Nsight Systems 的命令行工具，用于启动性能分析会话。
2. --stats=true
--stats=true：启用统计信息收集。这将在分析会话结束后生成详细的统计报告，包括各种性能指标和统计数据。
3. -t cuda,osrt,nvtx
-t：指定要跟踪的事件类型。
cuda：跟踪 CUDA API 调用和 CUDA 内核执行。
osrt：跟踪操作系统运行时（OS Runtime）事件，如线程创建、销毁、上下文切换等。
nvtx：跟踪 NVIDIA Tools Extension (NVTX) 标记和范围。NVTX 是一种用于在应用程序中插入标记和范围的工具，以便在性能分析时更容易识别关键代码段。
4. -o relu.prof
-o：指定输出文件名。
relu.prof：生成的性能分析报告文件名。Nsight Systems 将生成一个包含分析数据的 .qdrep 文件，以及一个可选的 .sqlite 数据库文件
#### ncu

### 尝试结果记录

| name  | name | 8192*8192 | 4096*4096 | 1024*1024 | 16384*16384 |
| ----- | -----| -------- | --------| --------- | -------| 
| naive | relu | 0.335667 ms | 0.041677ms | 0.004813ms | 1.319322ms |
| f16x2 | relu | 0.322355 ms | 0.025293ms | 0.003379ms | 1.287578ms |
| unpack | relu |  0.284672 ms | 0.043520ms | 0.003379ms | 0.000669ms |
| pack | relu | 0.285082 ms | 0.018944ms | 0.003174ms | 0.000688ms |

疑问：为什么后两种优化在大size下快这么多？

实验得到的现象：8196*8196还是正常的，但是8200*8200就开始不正常了。（这是为什么呢？）

## 