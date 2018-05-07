#include <iostream>
#include "softmaxlayer.h"
#include "softmaxlosslayer.h"

using namespace std;

int main()
{
    int num = 3;
    int channels = 5;
//    SoftmaxLayer softmaxLayer(num, channels);
    SoftmaxLossLayer softmaxlossLayer(num, channels);
    FP input[15] = {1, 6, 3, 2, 0, 8, 1, 0, 1, 3, 4, 5, 3, 0, 1};
    int labels[3] = {1, 0, 2};
    FP loss;
    FP* bottom_diff;
    softmaxlossLayer.forward(input, labels, loss);
    cout << "loss: " << loss << endl;
    softmaxlossLayer.backward(bottom_diff);
    cout << "bottom_diff: " << endl;
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < channels; j++) {
            cout << bottom_diff[i * channels + j] << " ";
        }
        cout << endl;
    }

}
