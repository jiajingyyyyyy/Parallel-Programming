#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>
#include <algorithm> // for std::swap
#include <iomanip>

const double PI = std::acos(-1.0);

// 位反转置换（Bit-reversal Permutation）,请补充
void bit_reverse(std::vector<std::complex<double>> &data)
{
    int n = data.size();
    if (n <= 1)
        return;

    // 计算表示 n 所需的位数
    int log_n = 0;
    while ((1 << log_n) < n)
        log_n++;

    // 遍历所有元素，进行位反转交换
    for (int i = 0; i < n; ++i)
    {
        int rev = 0;
        for (int j = 0; j < log_n; ++j)
        {
            if ((i >> j) & 1)
            {
                rev |= 1 << (log_n - 1 - j);
            }
        }

        if (i < rev)
        {
            std::swap(data[i], data[rev]);
        }
    }
}

// 并行 FFT（Cooley-Tukey，使用 OpenMP）,请补充
void fft_openmp(std::vector<std::complex<double>> &data)
{
    int n = data.size();
    if (n <= 1)
        return;

    bit_reverse(data);

    // 步骤 2: 蝶形运算，分阶段进行
    // m 是当前阶段的子 FFT 的大小，从 2, 4, 8, ... , n
    for (int m = 2; m <= n; m <<= 1)
    {
        // 计算当前阶段的基础旋转因子 W_m
        std::complex<double> wm = std::polar(1.0, -2.0 * PI / m);

// 并行化处理不同的数据块
// 每个线程处理一个或多个大小为 m 的数据块 (由 k 指定起始位置)
// 这些块之间没有数据依赖，可以安全地并行执行
#pragma omp parallel for
        for (int k = 0; k < n; k += m)
        {
            std::complex<double> w(1.0, 0.0); // 每个块的初始旋转因子 W_m^0
            // 对块内的每对元素进行蝶形运算
            for (int j = 0; j < m / 2; ++j)
            {
                // t = w * data[k + j + m / 2]
                std::complex<double> t = w * data[k + j + m / 2];
                // u = data[k + j]
                std::complex<double> u = data[k + j];
                // data[k + j] = u + t
                data[k + j] = u + t;
                // data[k + j + m / 2] = u - t
                data[k + j + m / 2] = u - t;
                // 更新旋转因子 w = w * wm
                w *= wm;
            }
        }
    }
}

// 串行 FFT（Cooley-Tukey，不使用 OpenMP）作为基准比较
void fft_serial(std::vector<std::complex<double>> &data)
{
    int n = data.size();
    if (n <= 1)
        return;

    // 步骤 1: 位反转置换
    bit_reverse(data);

    // 步骤 2: 蝶形运算，分阶段进行
    // m 是当前阶段的子 FFT 的大小，从 2, 4, 8, ... , n
    for (int m = 2; m <= n; m <<= 1)
    {
        // 计算当前阶段的基础旋转因子 W_m
        std::complex<double> wm = std::polar(1.0, -2.0 * PI / m);

        // 串行处理不同的数据块
        for (int k = 0; k < n; k += m)
        {
            std::complex<double> w(1.0, 0.0); // 每个块的初始旋转因子 W_m^0
            // 对块内的每对元素进行蝶形运算
            for (int j = 0; j < m / 2; ++j)
            {
                std::complex<double> t = w * data[k + j + m / 2];
                std::complex<double> u = data[k + j];
                data[k + j] = u + t;
                data[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
}

int main()
{
    int n;
    std::cin >> n;
    // 检查 n 是否为 2 的幂
    if (n <= 0 || (n & (n - 1)) != 0)
    {
        std::cerr << "Error: n must be a positive power of 2.\n";
        return 1;
    }

    std::vector<std::complex<double>> data(n);
    double real, imag;

    // 读取实部
    for (int i = 0; i < n; ++i)
    {
        std::cin >> real;
        data[i] = std::complex<double>(real, 0.0);
    }

    // 读取虚部
    for (int i = 0; i < n; ++i)
    {
        std::cin >> imag;
        data[i].imag(imag);
    }

    // 设置足够大的迭代次数，特别是对于小输入
    const int PARALLEL_ITERATIONS = 100;
    const int SERIAL_ITERATIONS = 10000; // 为串行计算设置更大的迭代次数

    // 创建数据副本
    std::vector<std::complex<double>> original_data = data;
    std::vector<std::complex<double>> data_parallel;
    std::vector<std::complex<double>> data_serial;

    // 预热缓存
    data_parallel = original_data;
    fft_openmp(data_parallel);
    data_serial = original_data;
    fft_serial(data_serial);

    // 执行并行 FFT
    data_parallel = original_data;
    double start_time = omp_get_wtime();
    for (int run = 0; run < PARALLEL_ITERATIONS; run++)
    {
        data_parallel = original_data;
        fft_openmp(data_parallel);
    }
    double parallel_time = omp_get_wtime() - start_time;
    parallel_time /= PARALLEL_ITERATIONS;

    // 执行串行 FFT - 使用更多迭代次数确保可测量
    data_serial = original_data;
    start_time = omp_get_wtime();
    for (int run = 0; run < SERIAL_ITERATIONS; run++)
    {
        data_serial = original_data;
        fft_serial(data_serial);
    }
    double serial_time = omp_get_wtime() - start_time;
    serial_time /= SERIAL_ITERATIONS;

    // 验证两种方法的结果是否一致
    bool correct = true;
    for (int i = 0; i < n; ++i)
    {
        if (std::abs(data_parallel[i].real() - data_serial[i].real()) > 1e-6 ||
            std::abs(data_parallel[i].imag() - data_serial[i].imag()) > 1e-6)
        {
            correct = false;
            std::cerr << "Results differ at index " << i << ": "
                      << data_parallel[i] << " vs " << data_serial[i] << std::endl;
            break;
        }
    }

    // 输出性能和验证信息，保持一致的精度格式
    std::cerr << "Serial execution time: " << std::fixed << std::setprecision(8) << serial_time << " seconds\n";
    std::cerr << "Parallel execution time: " << std::fixed << std::setprecision(8) << parallel_time << " seconds\n";

    // 计算加速比，支持更精确的小数
    double speedup = 0.0;
    if (serial_time > 1e-12)
    {
        speedup = serial_time / parallel_time;
    }

    // 对于小输入，并行版本可能会因为线程创建开销而变慢
    if (speedup < 1.0 && n < 1024)
    {
        std::cerr << "Speedup: " << std::fixed << std::setprecision(4) << speedup << "x (for small inputs, parallel overhead may exceed benefits)\n";
    }
    else
    {
        std::cerr << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }
    std::cerr << "Verification: " << (correct ? "PASSED" : "FAILED") << "\n";

    // 输出结果，保留6位小数
    for (int i = 0; i < n; ++i)
    {
        printf("%.6f %.6f\n", data_parallel[i].real(), data_parallel[i].imag());
    }

    return 0;
}