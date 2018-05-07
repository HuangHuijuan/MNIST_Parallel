#include <iostream>
#include <cuda.h>
#include "innerproductlayer.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        cout << "Usage: test_fc <0 for cpu 1 for gpu>" << endl;
        exit(-1);
    }
    bool gpu = atoi(argv[1]);
    if (gpu) {
        int gpucount = 0;
        int errorcode = cudaGetDeviceCount(&gpucount);
        if (errorcode == cudaErrorNoDevice) {
          cout << "No GPUs are visible" << std::endl;
          exit(-1);
        }
        int gpunum = 0;
        cudaSetDevice(gpunum);
        cout << "Using device " << gpunum << endl;;
    }
    FP lr = 0.001;
    int num = 2;
    int channels_in = 3;
    int channels_out = 4;
    FP input[6] = {2.1, 3.5, 1.4, 6.1, 4.5, 0.2};
    FP weights[12] = {1, 0.3, 0.8, 1.2, -0.3, -1.1, 0.8, 0.5, 1.4, 1.6, -0.7, 0.1};
    FP bias[4] = {0.1, 0.2, 0.3, 0.4};
    FP *output;
    InnerProductLayer innerProductLayer(gpu, num, channels_in, channels_out, lr, weights, bias);
    innerProductLayer.forward(input, output);

    cout << "output: " << endl;
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < channels_out; j++) {
            cout << output[i * channels_out + j] << " ";
        }
        cout << endl;
    }
    FP top_diff[8] = {1.2, -0.8, 0.7, -0.1, -0.3, 0.5, 0.6, -0.2};
    FP* bottom_diff;
    innerProductLayer.backward(top_diff, bottom_diff);
    cout << "bottom_diff: " << endl;
    for (int n = 0; n < num; n++) {
        for (int i =0; i < channels_in; i++) {
            cout << bottom_diff[n * channels_in + i] << " ";
        }
        cout << endl;
    }

}

