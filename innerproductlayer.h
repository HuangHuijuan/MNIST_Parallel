#ifndef INNERPRODUCTLAYER_H
#define INNERPRODUCTLAYER_H
#define FP float

class InnerProductLayer
{
public:
    InnerProductLayer(const bool gpu, const int num, const int channels_in, const int channels_out, const FP lr,
                      const FP* weights, const FP* bias);
    ~InnerProductLayer();

    void forward(const FP* input, FP* &output);
    void backward(const FP* top_diff, FP* &bottom_diff);

    bool gpu;
    int count;
    int num;
    int channels_in;
    int channels_out;
    FP lr;

    FP* weights;
    FP* bias;
    FP* dev_a, *dev_b, *dev_c;
    const FP* input;
};

#endif // INNERPRODUCTLAYER_H
