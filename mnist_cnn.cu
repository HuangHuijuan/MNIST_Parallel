#include <iostream>
#include <random>
#include <chrono>
#include "mnistdatalayer.h"
#include "convlayer.h"
#include "relulayer.h"
#include "innerproductlayer.h"
#include "softmaxlayer.h"
#include "softmaxlosslayer.h"

using namespace std;

void init_weights(FP* weights, int size) {
    std::default_random_engine generator;
    std::normal_distribution<FP> distribution(0.0,0.1);
    for (int i = 0; i < size; i++) {
        weights[i] = distribution(generator);
    }
}

void init_bias(FP* bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.1;
    }
}

void evaluate_model(MNISTDataLayer& dataLayer, ConvLayer& conv1, ReLULayer& relu1,
                    ConvLayer& conv2, ReLULayer& relu2, InnerProductLayer& fc,
                    SoftmaxLayer& softmaxLayer, bool gpu);

int main(int argc, char *argv[]) {
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
        cout << "Using device " << gpunum << endl;
    }

    int iterations = 10000;
    FP learning_rate = 0.001;
    int batch_size = 50;
    int im_h = 28;
    int im_w = 28;
    int conv1_h_out = 14;
    int conv1_w_out = 14;
    int conv1_channel_in = 1;
    int conv1_num_out = 20;
    int conv1_kernel = 5;
    int conv1_stride = 2;
    int conv1_pad = 2;
    int conv2_h_out = 7;
    int conv2_w_out = 7;
    int conv2_num_out = 50;
    int conv2_kernel = 5;
    int conv2_stride = 2;
    int conv2_pad = 2;
    int fc_channels_in = conv2_num_out * conv2_h_out * conv2_w_out;
    int num_classes = 10;

    int conv1_weights_size = conv1_num_out * conv1_channel_in * conv1_kernel * conv1_kernel;
    FP* conv1_weights = new FP[conv1_weights_size];
    init_weights(conv1_weights, conv1_weights_size);
    FP* conv1_bias = new FP[conv1_num_out];
    init_bias(conv1_bias, conv1_num_out);

    int conv2_weights_size = conv2_num_out * conv1_num_out * conv2_kernel * conv2_kernel;
    FP* conv2_weights = new FP[conv2_weights_size];
    init_weights(conv2_weights, conv2_weights_size);
    FP* conv2_bias = new FP[conv2_num_out];
    init_bias(conv2_bias, conv2_num_out);

    FP* fc_weights = new FP[fc_channels_in * num_classes];
    init_weights(fc_weights, fc_channels_in * num_classes);
    FP* fc_bias = new FP[num_classes];
    init_bias(fc_bias, num_classes);

    MNISTDataLayer dataLayer;
    ConvLayer conv1(gpu, batch_size, im_h, im_w, conv1_channel_in, conv1_num_out, conv1_kernel, conv1_kernel,
                    conv1_pad, conv1_pad, conv1_stride, conv1_stride, learning_rate, conv1_weights, conv1_bias);
    ReLULayer relu1(batch_size * conv1_num_out * conv1_h_out * conv1_w_out);
    ConvLayer conv2(gpu,batch_size, conv1_h_out, conv1_w_out, conv1_num_out, conv2_num_out, conv2_kernel, conv2_kernel,
                    conv2_pad, conv2_pad, conv2_stride, conv2_stride, learning_rate, conv2_weights, conv2_bias);
    ReLULayer relu2(batch_size * conv2_num_out * conv2_h_out * conv2_w_out);
    InnerProductLayer fc(gpu,batch_size, fc_channels_in, num_classes, learning_rate, fc_weights, fc_bias);
    SoftmaxLossLayer lossLayer(batch_size, num_classes);
    SoftmaxLayer softmaxLayer(batch_size, num_classes);

    FP loss;

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        FP *train_im_data;
        int *train_labels;
        FP *conv1_out;
        FP *relu1_out;
        FP *conv2_out;
        FP *relu2_out;
        FP *fc_out;
        FP *loss_diff;
        FP *fc_diff;
        FP *relu2_diff;
        FP *conv2_diff;
        FP *relu1_diff;
        FP *conv1_diff;

        //get training data
        dataLayer.get_train_data(batch_size, train_im_data, train_labels);

        //forward
        conv1.forward(train_im_data, conv1_out);
        relu1.forward(conv1_out, relu1_out);
        conv2.forward(relu1_out, conv2_out);
        relu2.forward(conv2_out, relu2_out);
        fc.forward(relu2_out, fc_out);
        lossLayer.forward(fc_out, train_labels, loss);
        if (i % 100 == 0) {
            cout << "[" << i << "]" << "loss: " << loss << endl;
        }
//        backward
        lossLayer.backward(loss_diff);
        fc.backward(loss_diff, fc_diff);
        relu2.backward(fc_diff, relu2_diff);
        conv2.backward(relu2_diff, conv2_diff);
        relu1.backward(conv2_diff, relu1_diff);
        conv1.backward(relu1_diff, conv1_diff);

        if (i != 0 && i % 1000 == 0) {
            evaluate_model(dataLayer, conv1, relu1, conv2, relu2, fc, softmaxLayer, gpu);
        }

        delete[] train_im_data;
        delete[] train_labels;
        delete[] conv1_out;
        delete[] relu1_out;
        delete[] conv2_out;
        delete[] relu2_out;
        delete[] fc_out;
        delete[] loss_diff;
        delete[] fc_diff;
        delete[] relu2_diff;
        delete[] conv2_diff;
        delete[] relu1_diff;
        delete[] conv1_diff;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << endl;

    evaluate_model(dataLayer, conv1, relu1, conv2, relu2, fc, softmaxLayer, gpu);

    delete[] conv1_weights;
    delete[] conv1_bias;
    delete[] conv2_weights;
    delete[] conv2_bias;
    delete[] fc_weights;
    delete[] fc_bias;

}

void evaluate_model(MNISTDataLayer& dataLayer, ConvLayer& conv1, ReLULayer& relu1,
                    ConvLayer& conv2, ReLULayer& relu2, InnerProductLayer& fc,
                    SoftmaxLayer& softmaxLayer, bool gpu)
{
    FP *test_im_data;
    int *test_labels;
    FP *conv1_out;
    FP *relu1_out;
    FP *conv2_out;
    FP *relu2_out;
    FP *fc_out;

    dataLayer.get_test_data(10000, test_im_data, test_labels);
    conv1.forward(test_im_data, conv1_out);
    relu1.forward(conv1_out, relu1_out);
    conv2.forward(relu1_out, conv2_out);
    relu2.forward(conv2_out, relu2_out);
    fc.forward(relu2_out, fc_out);

    FP accuracy = softmaxLayer.get_accuracy(fc_out, test_labels);
    cout << "Accuracy: " << accuracy << endl;

    delete[] test_im_data;
    delete[] test_labels;
    delete[] conv1_out;
    delete[] relu1_out;
    delete[] conv2_out;
    delete[] relu2_out;
    delete[] fc_out;
}
