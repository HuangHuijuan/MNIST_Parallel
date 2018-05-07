#include <algorithm>
#include "relulayer.h"

ReLULayer::ReLULayer(int count)
    :count(count)
{

}

ReLULayer::~ReLULayer()
{
}

void ReLULayer::forward(const FP* input, FP* &output)
{
    this->input = input;
    output = new FP[count];
    for (int i = 0; i < count; i++) {
        output[i] = std::max(FP(0), input[i]);
    }
}

void ReLULayer::backward(const FP* top_diff, FP* &bottom_diff)
{
    bottom_diff = new FP[count];
    for (int i = 0; i < count; i++) {
        bottom_diff[i] = input[i] >= 0? top_diff[i]: 0.;
    }
}
