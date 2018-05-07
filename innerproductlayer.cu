#include <iostream>
#include <stdio.h>
#include "innerproductlayer.h"
#include "utils.h"
using namespace std;

InnerProductLayer::InnerProductLayer(const bool gpu, const int num, const int channels_in, const int channels_out,
                                     const FP lr, const FP* weights, const FP* bias)
    :gpu(gpu), num(num), channels_in(channels_in), channels_out(channels_out), lr(lr)
{
    int weights_size = channels_out * channels_in;
    this->weights = new FP[weights_size];
    memcopy_cpu(this->weights, weights, weights_size * sizeof(FP));
    this->bias = new FP[channels_out];
    memcopy_cpu(this->bias, bias, channels_out * sizeof(FP));
    count = num * channels_in;
    if (gpu) {
        cudaMalloc((void**)&dev_a, num * channels_in * sizeof(FP));
        cudaMalloc((void**)&dev_b, channels_in * channels_out * sizeof(FP));
        cudaMalloc((void**)&dev_c, num * channels_out * sizeof(FP));
    }
}

InnerProductLayer::~InnerProductLayer()
{
    delete[] weights;
    delete[] bias;
    if (gpu) {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
}

void InnerProductLayer::forward(const FP *input, FP *&output)
{
    this->input = input;
    output = new FP[num * channels_out];
    for (int i = 0; i < num * channels_out; i++)
        output[i] = 0;

    if (gpu) {
        matrixmult_gpu(false, true, input, weights, output, num, channels_out, channels_in, dev_a, dev_b, dev_c);
    } else {
        matrixmult_btrans_cpu(input, weights, output, num, channels_out, channels_in);
    }
    for (int i = 0; i < num; i++) {
        add_bias_cpu(output + i * channels_out, bias, 1, channels_out);
    }

}

void InnerProductLayer::backward(const FP *top_diff, FP *&bottom_diff)
{
    bottom_diff = new FP[num * channels_in];
    for (int i = 0; i < num * channels_in; i++)
        bottom_diff[i] = 0;
    int weights_size = channels_out * channels_in;
    FP* weights_diff = new FP[weights_size];
    FP* weights_diff_tmp = new FP[weights_size];
    for (int i = 0; i < weights_size; i++) {
        weights_diff[i] = 0;
        weights_diff_tmp[i] = 0;
    }
    FP* bias_diff = new FP[channels_out];
    for (int i = 0; i < channels_out; i++) {
        bias_diff[i] = 0;
    }

    if (gpu) {
        //update bottom diff
        matrixmult_gpu(false, false, top_diff, weights, bottom_diff, num, channels_in, channels_out, dev_c, dev_b, dev_a);
        //update weights diff
        matrixmult_gpu(true, false, top_diff, input, weights_diff_tmp, channels_out, channels_in, num, dev_c, dev_a, dev_b);
        for (int j = 0; j < weights_size; j++) {
            weights_diff[j] += weights_diff_tmp[j];
        }
        //update bias diff
        for (int i = 0; i < num; i++) {
            for (int c = 0; c < channels_out; c++) {
                bias_diff[c] += top_diff[i * channels_out + c];
            }
        }
    } else {
        //update bottom diff
        matrixmult_cpu(top_diff, weights, bottom_diff, num, channels_in, channels_out);
        //update weights diff
        matrixmult_atrans_cpu(top_diff, input, weights_diff, channels_out, channels_in, num);
        //update bias diff
        for (int i = 0; i < num; i++) {
            for (int c = 0; c < channels_out; c++) {
                bias_diff[c] += top_diff[i * channels_out + c];
            }
        }
    }

//    cout << "weights_diff: " << endl;
//    for (int i =0; i < channels_out; i++) {
//        for (int c = 0; c < channels_in; c++) {
//           cout << weights_diff[i * channels_in + c] << " ";
//        }
//        cout << endl;
//    }
//    cout << "bias_diff: " << endl;
//    for (int i = 0; i < channels_out; i++) {
//        cout << bias_diff[i] <<endl;
//    }

    //update weights
    update_params_cpu(weights, weights_diff, weights_size, num, lr);

    //update bias
    update_params_cpu(bias, bias_diff, channels_out, num, lr);
}

