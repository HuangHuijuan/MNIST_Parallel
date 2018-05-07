#include <math.h>
#include <iostream>
#include <cfloat>
#include <algorithm>
#include "softmaxlosslayer.h"
using namespace std;

SoftmaxLossLayer::SoftmaxLossLayer(const int num, const int channles)
    :num(num), channles(channles)
{
    softmaxlayer = new SoftmaxLayer(num, channles);
}

SoftmaxLossLayer::~SoftmaxLossLayer()
{
    delete softmaxlayer;
}

void SoftmaxLossLayer::forward(const FP* input, const int* labels, FP &loss)
{
    this->labels = labels;
    softmaxlayer->forward(input, prob_data);

    for (int i = 0; i < num; i++) {
        if (labels[i] < 0 || labels[i] > 9) {
            cout << "invalid label " << labels[i] << endl;
        }
        loss -= log(max(prob_data[i * channles + labels[i]], FP(FLT_MIN)));
    }
    loss /= num;
}

void SoftmaxLossLayer::backward(FP* &bottom_diff)
{
    bottom_diff = new FP[num * channles];
    for (int i = 0; i < num * channles; i++) {
        bottom_diff[i] = prob_data[i];
    }
    for (int i = 0; i < num; i++) {
        bottom_diff[i * channles + labels[i]] -= 1;
    }
    for (int i = 0; i < num * channles; i++) {
        bottom_diff[i] /= num;
    }
}
