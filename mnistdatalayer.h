#ifndef MNISTDATALAYER_H
#define MNISTDATALAYER_H
#define FP float
#include <string>
#include <vector>


class MNISTDataLayer
{
public:
    MNISTDataLayer();
    ~MNISTDataLayer();

    void load_mnist(std::string dir);
    void get_train_data(const int batch_size, FP* &batch_im_data, int* &batch_labels);
    void get_test_data(const int batch_size, FP* &batch_im_data, int* &batch_labels);

    uint32_t swap_endian(uint32_t val);

    const int MEAN = 127;
    int train_num_images;
    int test_num_images;
    FP* train_data_im;
    int* train_labels;
    FP* test_data_im;
    int* test_labels;
    std::vector<int> train_indexes;
    int train_idx;
};

#endif // DATALAYER_H
