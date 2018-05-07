#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H
#define FP float

class SoftmaxLayer
{
public:
    SoftmaxLayer(const int num, const int channels);
    ~SoftmaxLayer();

    void forward(const FP* input, FP* &output);

    FP get_accuracy(const FP* input, const int* labels);

    int num;
    int channels;

};

#endif // SOFTMAXLAYER_H
