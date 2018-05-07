#include <iostream>
#include "utils.h"
#include "convlayer.h"
#define Block_Dim 32
#define NUM_THREADS Block_Dim * Block_Dim
using namespace std;

ConvLayer::ConvLayer(bool gpu, const int num, const int height, const int width, const int channels_in, const int channels_out,
                     const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_w, const int stride_h,
                     const FP lr, const FP* weights, const FP* bias)
    : gpu(gpu), num(num), height(height), width(width), channels_in(channels_in), channels_out(channels_out),
      kernel_h(kernel_h), kernel_w(kernel_w), pad_h(pad_h), pad_w(pad_w), stride_w(stride_w), stride_h(stride_h), lr(lr)
{
    height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1; //the width of output feature maps
    width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1; //the height of output feature maps
    int weights_size = channels_out * channels_in * kernel_h * kernel_w;
    this->weights = new FP[weights_size];
    memcopy_cpu(this->weights, weights, weights_size * sizeof(FP));
    this->bias = new FP[channels_out];
    memcopy_cpu(this->bias, bias, channels_out * sizeof(FP));
    count = num * channels_in * height * width;

    //allocate gpu memory for im2col and col2im
    num_kernels_im2col = channels_in * height_out * width_out;
    num_kernels_col2im = channels_in * height * width;

    num_blocks_im2col = (num_kernels_im2col + NUM_THREADS - 1) / NUM_THREADS;
    num_blocks_col2im = (num_kernels_col2im + NUM_THREADS - 1) / NUM_THREADS;
    kernel_size = channels_in * kernel_h * kernel_w;
    map_size = height_out * width_out;

    data_im_size = height * width * channels_in * sizeof(FP);
    data_col_size = kernel_size * map_size * sizeof(FP);
    if (gpu) {
        cudaMalloc((void**)&dev_data_im, data_im_size);
        cudaMalloc((void**)&dev_data_col, data_col_size);
    }

    //allocate gpu memory for matrixmul
    if (gpu) {
        cudaMalloc((void**)&dev_a, channels_out * kernel_size * sizeof(FP));
        cudaMalloc((void**)&dev_b, kernel_size * map_size * sizeof(FP));
        cudaMalloc((void**)&dev_c, channels_out * map_size * sizeof(FP));
        cudaMalloc((void**)&dev_d, channels_out * sizeof(FP));
        cudaMalloc((void**)&dev_e, map_size * sizeof(FP));
    }
}

ConvLayer::~ConvLayer()
{
    delete[] weights;
    delete[] bias;
    //free the allocated device global memory
    if (gpu) {
        cudaFree(dev_data_im);
        cudaFree(dev_data_col);
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaFree(dev_d);
        cudaFree(dev_e);
    }
}

void ConvLayer::forward(const FP* input, FP* &output)
{
    this->input = input;

    int out_col_offset = channels_out * map_size;

    int im_offset = channels_in * height * width;
    int output_size = num * channels_out * height_out * width_out;
    output = new FP[output_size];
    for (int i = 0; i < output_size; i++) output[i] = 0;
    FP* im_col = new FP[kernel_size * map_size];

    for (int i = 0; i < num; i++) {
        if (gpu) {

            im2col_gpu(channels_in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, input + i * im_offset, dev_data_im,
                       im_col, dev_data_col, data_im_size, data_col_size, height_out, width_out, num_kernels_im2col, num_blocks_im2col);
            matrixmult_gpu(false, false, weights, im_col, output + i * out_col_offset, channels_out, map_size, kernel_size,
                           dev_a, dev_b, dev_c);
        } else {
            im2col_cpu(input + i * im_offset, channels_in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, im_col);
            matrixmult_cpu(weights, im_col, output + i * out_col_offset, channels_out, map_size, kernel_size);
        }
        add_bias_cpu(output + i * out_col_offset, bias, map_size, channels_out);
    }

    delete[] im_col;
}

void ConvLayer::backward(const FP* top_diff, FP* &bottom_diff)
{

    int bottom_diff_offset = channels_in * height * width;
    bottom_diff = new FP[num * bottom_diff_offset];
    for (int i = 0; i < num * bottom_diff_offset; i++) bottom_diff[i] = 0;

    int top_diff_offset = channels_out * height_out * width_out;

    int input_offset = channels_in * height * width;

    int weights_size = channels_out * kernel_size;
    FP* weights_diff = new FP[weights_size];
    FP* weights_diff_tmp = new FP[weights_size];
    for (int i = 0; i < weights_size; i++) {
        weights_diff[i] = 0;
        weights_diff_tmp[i] = 0;
    }

    FP* bias_diff = new FP[channels_out];
    FP* bias_diff_tmp = new FP[channels_out];
    for (int i = 0; i < channels_out; i++) {
        bias_diff[i] = 0; //(channels_out, 1)
        bias_diff_tmp[i] = 0;
    }

    FP* bias_multiplier = new FP[map_size];
    for (int i = 0; i < map_size; i++) bias_multiplier[i] = 1; //(1, map_size)

    FP* out_col = new FP[kernel_size * map_size];
    FP* input_col = new FP[kernel_size * map_size];

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < kernel_size * map_size; j++) {
            out_col[j] = 0;
            input_col[j] = 0;
        }
        if (gpu) {
            //calculate bottom diff, which will be propogated to the previous layer
            matrixmult_gpu(true, false, weights, top_diff + i * top_diff_offset, out_col, kernel_size, map_size, channels_out, dev_a, dev_c, dev_b);

            col2im_gpu(channels_in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, out_col, dev_data_col,
                       bottom_diff + i * bottom_diff_offset, dev_data_im, data_im_size, data_col_size, height_out, width_out,
                       num_kernels_col2im, num_blocks_col2im);

            //calculate weights diff
            im2col_gpu(channels_in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, input + i * input_offset, dev_data_im,
                       input_col, dev_data_col, data_im_size, data_col_size, height_out, width_out, num_kernels_im2col, num_blocks_im2col);
            matrixmult_gpu(false, true, top_diff + i * top_diff_offset, input_col, weights_diff_tmp, channels_out, kernel_size, map_size, dev_c, dev_b, dev_a);
            for (int j = 0; j < weights_size; j++) {
                weights_diff[j] += weights_diff_tmp[j];
            }
            //update bias diff
            matrixmult_gpu(false, true, top_diff + i * top_diff_offset, bias_multiplier, bias_diff_tmp, channels_out, 1, map_size, dev_c, dev_e, dev_d);
            for (int j = 0; j < channels_out; j++) {
                bias_diff[j] += bias_diff_tmp[j];
            }
        } else {
            //calculate bottom diff, which will be propogated to the previous layer
            matrixmult_atrans_cpu(weights, top_diff + i * top_diff_offset, out_col, kernel_size, map_size, channels_out);
            col2im_cpu(out_col, channels_in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, bottom_diff + i * bottom_diff_offset);

            //calculate weights diff
            im2col_cpu( input + i * input_offset, channels_in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, input_col);
            matrixmult_btrans_cpu(top_diff + i * top_diff_offset, input_col, weights_diff, channels_out, kernel_size, map_size);

            //update bias diff
            matrixmult_btrans_cpu(top_diff + i * top_diff_offset, bias_multiplier, bias_diff, channels_out, 1, map_size);
        }

    }

    delete[] out_col;
    delete[] input_col;
    delete[] weights_diff_tmp;
    delete[] bias_diff_tmp;

     //update weights
     update_params_cpu(weights, weights_diff, weights_size, num, lr);
     //update bias
     update_params_cpu(bias, bias_diff, channels_out, num, lr);
}

