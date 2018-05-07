#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include "mnistdatalayer.h"
using namespace std;

MNISTDataLayer::MNISTDataLayer()
{
    train_num_images = 60000;
    test_num_images = 10000;
    train_idx = 0;
    train_data_im = new FP[train_num_images * 784];
    test_data_im = new FP[test_num_images * 784];
    train_labels = new int[train_num_images];
    test_labels = new int[test_num_images];

    load_mnist("mnist_data/");

    for (int i = 0; i < train_num_images; i++) {
        train_indexes.push_back(i);
    }
    shuffle(train_indexes.begin(), train_indexes.end(), std::default_random_engine(0));
}

MNISTDataLayer::~MNISTDataLayer()
{
    delete[] train_data_im;
    delete[] test_data_im;
    delete[] train_labels;
    delete[] test_labels;

}

void MNISTDataLayer::load_mnist(string dir)
{
    unsigned char pixels[784];  // 28 * 28
    unsigned char labels_buff[1000];

    uint32_t magic, num_images, rows, cols, num_labels;

    string train_image_path = dir + "train-images.idx3-ubyte";
    string train_label_path = dir + "train-labels.idx1-ubyte";
    string test_image_path = dir + "t10k-images.idx3-ubyte";
    string test_label_path = dir + "t10k-labels.idx1-ubyte";

    std::ifstream file;

    cout << "train_image_path: " << train_image_path << endl;
    //load train images
    file.open(train_image_path, std::ios::in | std::ios::binary);
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) {
        cout << "incorrect magic number " << magic << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&num_images), 4);
    num_images = swap_endian(num_images);
    if (num_images != train_num_images) {
        cout << "incorrect images number " << num_images << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    if (rows != 28) {
        cout << "incorrect rows" << rows << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);
    if (cols != 28) {
        cout << "incorrect cols" << cols << endl;
        exit(0);
    }

    for (int idx = 0; idx < num_images; idx++){
         file.read(reinterpret_cast<char*>(pixels), 784);
         for (int i = 0; i < 784; i++){
              train_data_im[idx * 784 + i] = (FP)pixels[i] - MEAN;
         }
    }

    file.close();

    //load train labels
    file.open(train_label_path, std::ios::in | std::ios::binary);
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        cout << "incorrect magic number " << magic << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if (num_labels != train_num_images) {
        cout << "incorrect labels number " << num_labels << endl;
        exit(0);
    }

    for(int idx = 0; idx < 60; idx++){
        file.read(reinterpret_cast<char*>(labels_buff), 1000);
        for(int i = 0; i < 1000; i++){
            int count = idx * 1000 + i;
            if(count < num_labels){
                train_labels[count] = (int)labels_buff[i];
            }
        }
    }
    file.close();

    //load test images
    file.open(test_image_path, std::ios::in | std::ios::binary);
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) {
        cout << "incorrect magic number " << magic << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&num_images), 4);
    num_images = swap_endian(num_images);
    if (num_images != test_num_images) {
        cout << "incorrect images number " << num_images << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    if (rows != 28) {
        cout << "incorrect rows" << rows << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);
    if (cols != 28) {
        cout << "incorrect cols" << cols << endl;
        exit(0);
    }

    for (int idx = 0; idx < num_images; idx++){
         file.read(reinterpret_cast<char*>(pixels), 784);
         for (int i = 0; i < 784; i++){
              test_data_im[idx * 784 + i] = (FP)pixels[i] - MEAN;
         }
    }

    file.close();

    //load train labels
    file.open(test_label_path, std::ios::in | std::ios::binary);
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        cout << "incorrect magic number " << magic << endl;
        exit(0);
    }
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if (num_labels != test_num_images) {
        cout << "incorrect labels number " << num_labels << endl;
        exit(0);
    }

    for(int idx = 0; idx < 60; idx++){
        file.read(reinterpret_cast<char*>(labels_buff), 1000);
        for(int i = 0; i < 1000; i++){
            int count = idx * 1000 + i;
            if(count < num_labels){
                test_labels[count] = (int)labels_buff[i];
            }
        }
    }

    file.close();
}

void MNISTDataLayer::get_train_data(const int batch_size, FP* &batch_im_data, int* &batch_labels)
{
    batch_im_data = new FP[batch_size * 784];
    batch_labels = new int[batch_size];
    for (int cnt = 0; cnt < batch_size; train_idx++, cnt++) {
        train_idx %= train_num_images;
        int idx = train_indexes[train_idx];
        for (int i = 0; i < 784; i++) {
            batch_im_data[cnt * 784 + i] = train_data_im[idx * 784 + i];
        }
        batch_labels[cnt] = train_labels[idx];
    }
}

void MNISTDataLayer::get_test_data(const int batch_size, FP* &batch_im_data, int* &batch_labels)
{
    batch_im_data = new FP[batch_size * 784];
    batch_labels = new int[batch_size];
    for (int cnt = 0; cnt < min(batch_size, test_num_images); cnt++) {
        for (int i = 0; i < 784; i++) {
            batch_im_data[cnt * 784 + i] = test_data_im[cnt * 784 + i];
        }
        batch_labels[cnt] = test_labels[cnt];
    }
}

uint32_t MNISTDataLayer::swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
}
