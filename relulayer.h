#ifndef RELULAYER_H
#define RELULAYER_H
#define FP float

class ReLULayer
{
public:
    ReLULayer(int count);
    ~ReLULayer();
    void forward(const FP* input, FP* &output);
    void backward(const FP* top_diff, FP* &bottom_diff);

    int count;
    const FP* input;
};

#endif // RELULAYER_H
