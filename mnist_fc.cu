#include <iostream>
#include <chrono>
#include "mnistdatalayer.h"
#include "innerproductlayer.h"
#include "softmaxlayer.h"
#include "softmaxlosslayer.h"

using namespace std;

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
        cout << "Using device " << gpunum << endl;;
    }

    float elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // instrument code to measure start time
    cudaEventCreate(&stop);

    int iterations = 5000;
    FP learning_rate = 0.1;
    int batch_size = 100;
    int channels_in = 784;
    int channels_out = 10;
    int dim = channels_in * channels_out;
    FP *weights = new FP[dim];
    for (int i = 0; i < dim; i++) {
        weights[i] = FP(0);
    }
    FP *bias = new FP[channels_out];
    for (int i = 0; i < channels_out; i++) {
        bias[i] = FP(0);
    }
    FP loss;

    MNISTDataLayer dataLayer;
    InnerProductLayer innerProductLayer(gpu, batch_size, channels_in, channels_out, learning_rate, weights, bias);
    SoftmaxLossLayer lossLayer(batch_size, channels_out);

    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
        FP *train_im_data;
        int *train_labels;
        FP *fc_out;
        FP *loss_diff;
        FP *fc_diff;

        //get training data
        dataLayer.get_train_data(batch_size, train_im_data, train_labels);

        innerProductLayer.forward(train_im_data, fc_out);
        lossLayer.forward(fc_out, train_labels, loss);
        if (i % 100 == 0) {
            cout << "[" << i << "]" << "loss: " << loss << endl;
        }

        //backward
        lossLayer.backward(loss_diff);
        innerProductLayer.backward(loss_diff, fc_diff);

        delete[] train_im_data;
        delete[] train_labels;
        delete[] fc_out;
        delete[] loss_diff;
        delete[] fc_diff;
    }
    cudaEventRecord(stop, 0); // instrument code to measure end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop );
    cout << "Train Time: " << elapsed_time_ms << endl;

    FP *test_fc_out;
    FP *test_im_data;
    int *test_labels;
    SoftmaxLayer softmaxLayer(batch_size, channels_out);

    dataLayer.get_test_data(10000, test_im_data, test_labels);
    innerProductLayer.forward(test_im_data, test_fc_out);
    FP accuracy = softmaxLayer.get_accuracy(test_fc_out, test_labels);
    cout << "Test accuracy: " << accuracy << endl;

    delete[] test_im_data;
    delete[] test_labels;
    delete[] weights;
    delete[] bias;
    delete[] test_fc_out;

}
