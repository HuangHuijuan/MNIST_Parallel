#include <iostream>
#include <cuda.h>
#include <chrono>
#include <random>
#include "utils.h"
#define FP float

using namespace std;

int main() {
    float elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // instrument code to measure start time
    cudaEventCreate(&stop);
//    int n = 1024;
//    int m = 1024;
//    int p = 1024;
//    FP* a = new FP[n * p];
//    for (int i = 0; i < n * p; i++) {
//        a[i] = static_cast <FP> (rand()) / static_cast <FP> (RAND_MAX);
//    }
//    FP* b = new FP[p * m];
//    for (int i = 0; i < p * m; i++) {
//        b[i] = static_cast <FP> (rand()) / static_cast <FP> (RAND_MAX);
//    }
//    FP *c = new FP[n * m];
//    for (int i = 0; i < n * m; i++) {
//        c[i] = 0;
//    }

//    auto t1 = std::chrono::high_resolution_clock::now();
//    matrixmult_gpu(false, true, a, b, c, n, m, p);
//    auto t2 = std::chrono::high_resolution_clock::now();
//    cout << "GPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << endl;

//    t1 = std::chrono::high_resolution_clock::now();
//    matrixmult_btrans_cpu(a, b, c, n, m, p);
//    t2 = std::chrono::high_resolution_clock::now();
//    cout << "CPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << endl;

    int channels = 1000;
    int height = 100;
    int width = 100;
    int kernel_h = 5;
    int kernel_w = 5;
    int pad_h = 2;
    int pad_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int height_col =  (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    int num_blocks = (num_kernels + 1024 - 1) / 1024;


    int size = channels * height * width;
    FP* data_im = new FP[size];
    int col_size = channels * kernel_h * kernel_w * height_col * width_col;
    FP *data_col1 = new FP[col_size];
    FP *data_col2 = new FP[col_size];
    for (int i = 0; i < size; i++) {
        data_im[i] = static_cast <FP> (rand()) / static_cast <FP> (RAND_MAX);
    }

    FP *dev_data_im, *dev_data_col;
    cudaMalloc((void**)&dev_data_im, size * sizeof(FP));
    cudaMalloc((void**)&dev_data_col, col_size * sizeof(FP));

    cudaEventRecord(start, 0);
    im2col_gpu(channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_im, dev_data_im,
               data_col1, dev_data_col, size * sizeof(FP), col_size * sizeof(FP), height_col, width_col, num_kernels, num_blocks);

    cudaEventRecord(stop, 0); // instrument code to measure end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop );
    cout << "GPU Time: " << elapsed_time_ms << endl;

    cudaEventRecord(start, 0);
    im2col_cpu(data_im, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,data_col2);
    cudaEventRecord(stop, 0); // instrument code to measure end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop );
    cout << "CPU Time: " << elapsed_time_ms << endl;

    FP error = 0;
    for (int i = 0; i < col_size; i++) {
        error += data_col1[i] - data_col2[i];
    }
    cout << "error: " << error << endl;

    delete[] data_im;
    delete[] data_col1;
    delete[] data_col2;
    cudaFree(dev_data_im);
    cudaFree(dev_data_col);
}
