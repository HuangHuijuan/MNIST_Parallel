#include <iostream>
#include "mnistdatalayer.h"

using namespace std;

int main()
{
    MNISTDataLayer datalayer;
    FP* im_data;
    int* labels;
    int batch_size = 10;
    datalayer.get_train_data(batch_size, im_data, labels);
    cout << "train labels: " << endl;
    for (int i = 0; i < batch_size; i++) {
        cout << labels[i] << " ";
    }
    cout << endl;

    cout << "The first train image:" << endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            cout << im_data[i * 28 + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << "test labels: " << endl;
    datalayer.get_test_data(batch_size, im_data, labels);
    for (int i = 0; i < batch_size; i++) {
        cout << labels[i] << " ";
    }
    cout << endl;
    cout << "The second test image:" << endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            cout << im_data[784 + i * 28 + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
