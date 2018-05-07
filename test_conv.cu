#include <iostream>
#include <cuda.h>
#include "convlayer.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        cout << "Usage: test_fc <0 for cpu 1 for gpu>" << endl;
        exit(-1);
    }
    bool gpu = atoi(argv[1]);
    FP lr = 0.001;
    int num = 2;
    int channels_in = 1;
    int channels_out = 1;
    int kernel_h = 2;
    int kernel_w = 2;
    int pad_h = 0;
    int pad_w = 0;
    int stride_w = 1;
    int stride_h = 1;
    int height = 3;
    int width = 3;
    int height_out = 2;
    int width_out = 2;

    FP* output;

//    int weights_size = out_channels * in_channels * kernel_h * kernel_w;
//    FP* weights = new FP[weights_size];
//    FP* bias = new FP[out_channels];
//    FP bias[2] = {1, 0};
//    FP weights[54] = {0,1,0,  1,-1,1,  0,-1,-1,  0,-1,-1,  0,1,1,  0,1,1,  -1,-1,1,  1,0,0,  -1,-1,-1,
//               1,1,0,  0,0,0,  0,0,0,  0,-1,1,  1,1,0,  0,1,0,  -1,0,0,  -1,0,1,  1,0,1};
    //    FP weights[8] = {
    //        1,1,1,1,1,1,1,1
    //    };

//    FP input[75] = {1,0,1,2,1,
//                   2,2,0,0,2,
//                   2,1,0,0,2,
//                   2,0,2,1,1,
//                   0,1,0,2,2,
//                  1,1,0,0,0,
//                   2,1,0,1,2,
//                   2,1,2,1,1,
//                   2,0,1,1,0,
//                   2,2,1,0,1,
//                  1,0,0,2,0,
//                   1,1,0,0,0,
//                   2,0,2,0,1,
//                   2,1,1,2,1,
//                   2,2,2,1,2};
    FP bias[1] = {0};
    FP weights[4] = {-1, 1, 1, 0};
    FP input[18] = {1, 2, 1, 0, 1, -1, 1, 0, 2,
                  -1, 1, 1, 0, 2, 1, -1, 1, 0};

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

    ConvLayer conv(gpu, num, height, width, channels_in, channels_out, kernel_h, kernel_w, pad_h, pad_w, stride_w, stride_h, lr, weights, bias);
    conv.forward(input, output);

    cout << "forward output:" << endl;
    for (int n = 0; n < num; n++) {
        for (int i = 0; i < channels_out; i++) {
            for (int j = 0; j < height_out; j++) {
                for (int k = 0; k < width_out; k++) {
                    cout << output[n * channels_out * height_out * width_out + i * height_out * width_out + j * width_out + k] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }


    FP top_diff[8] = {1, 0, -1, 1, 2, -1, 1, 0};
    FP* bottom_diff;

    conv.backward(top_diff, bottom_diff);

    cout << "bottom_diff: " << endl;
    for (int n = 0; n < num; n++) {
        for (int i =0; i < channels_in; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    cout << bottom_diff[n * channels_in * height * width + i * height * width + j * width + k] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }

//    int num = 1;
//    int in_channels = 2;
//    int out_channels = 1;
//    int kernel_h = 2;
//    int kernel_w = 2;
//    int pad_h = 0;
//    int pad_w = 0;
//    int stride_w = 2;
//    int stride_h = 2;
//    ConvLayer conv(num, in_channels, out_channels, kernel_h, kernel_w, pad_h, pad_w, stride_w, stride_h);
//    FP input[16] = {1,2,3,4,
//                      5,6,7,8,
//                      9,10,11,12,
//                      13,14,15,16
//                        };
//    int height = 2;
//    int width = 4;
//    FP* output = new FP[4];
//    conv.conv_forward_cpu(input, output, &height, &width);
//    for (int i = 0; i < 1; i++) {
//        for (int j = 0; j < 1; j++) {
//            for (int k = 0; k < 2; k++) {
//                cout << output[i * 2 + j * 3 + k] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
    delete[] output;
    return 0;
}
