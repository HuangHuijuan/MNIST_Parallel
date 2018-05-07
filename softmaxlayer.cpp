#include <algorithm>
#include <math.h>
#include "softmaxlayer.h"

SoftmaxLayer::SoftmaxLayer(const int num, const int channels)
    :num(num), channels(channels)
{

}

SoftmaxLayer::~SoftmaxLayer()
{

}

void SoftmaxLayer::forward(const FP *input, FP *&output)
{
    int size = num * channels;
    output = new FP[size];

    for (int i = 0; i < num; i++) {
        FP max_num = 0;
        FP sum = 0;
        for (int c = 0; c < channels; c++) {
            max_num = std::max(max_num, input[i * channels + c]);
        }
        for (int c = 0; c < channels; c++) {
            FP tmp = exp(input[i * channels + c] - max_num);
            output[i * channels + c] = tmp;
            sum += tmp;
        }
        for (int c = 0; c < channels; c++) {
            output[i * channels + c] /= sum;
        }
    }
}

FP SoftmaxLayer::get_accuracy(const FP *input, const int *labels)
{
    FP *prob_data;
    this->forward(input, prob_data);
    FP cnt = 0;

    for (int n = 0; n < num; n++) {
        int idx = -1;
        FP max_prob = 0;
        for (int c = 0; c < channels; c++) {
            FP prob = prob_data[n * channels + c];
            if (prob > max_prob) {
                max_prob = prob;
                idx = c;
            }
        }
        if (idx == labels[n]) {
            cnt++;
        }
    }
    return cnt / num;
}

