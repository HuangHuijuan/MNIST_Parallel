#ifndef SOFTMAXLOSSLAYER_H
#define SOFTMAXLOSSLAYER_H
#define FP float
#include "softmaxlayer.h"

class SoftmaxLossLayer
{
public:
    SoftmaxLossLayer(const int num, const int channles);
    ~SoftmaxLossLayer();

    void forward(const FP* input, const int* labels, FP &loss);
    void backward(FP* &bottom_diff);

    int num;
    int channles;
    FP* prob_data;
    const int* labels;
    SoftmaxLayer* softmaxlayer;
};

#endif // SOFTMAXLOSSLAYER_H
