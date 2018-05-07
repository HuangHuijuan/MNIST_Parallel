#include <iostream>
#include "relulayer.h"

using namespace std;

int main()
{
    int count = 6;
    FP *output;
    FP input[6] = {-1.2, 2, 0.3, 1.1, 0, -0.1};
    ReLULayer reluLayer(count);
    reluLayer.forward(input, output);
    for (int i = 0; i < count; i++)
        cout << output[i] << " ";
    cout << endl;
    FP top_diff[6] = {0.3, -0.1, 0.4, -0.1, 0.3, -0.4};
    FP *bottom_diff;
    reluLayer.backward(top_diff, bottom_diff);
    for (int i = 0; i < count; i++)
        cout << bottom_diff[i] << " ";
    cout << endl;
}
