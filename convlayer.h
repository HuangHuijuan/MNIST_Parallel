#ifndef CONVLAYER_H
#define CONVLAYER_H
#define FP float

class ConvLayer
{
public:
    ~ConvLayer();
    ConvLayer(bool gpu, const int num, const int height, const int width, const int channels_in, const int channels_out,
              const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_w, const int stride_h,
             const FP lr, const FP* weights, const FP* bias);

    void forward(const FP* input, FP* &output);
    void backward(const FP* top_diff, FP* &bottom_diff);

    bool gpu;
    int count;
    int num;
    int height;
    int width;
    int channels_in;
    int channels_out;
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int stride_w;
    int stride_h;
    FP lr;
    int height_out;
    int width_out;
    int num_kernels_im2col;
    int num_kernels_col2im;
    int num_blocks_im2col;
    int num_blocks_col2im;
    int data_im_size;
    int data_col_size;
    int kernel_size;
    int map_size;

    FP* dev_data_col, *dev_data_im;
    FP* weights;
    FP* bias;
    FP* dev_a, *dev_b, *dev_c, *dev_d, *dev_e;
    const FP* input;
};

#endif // CONVLAYER_H
