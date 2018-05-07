#ifndef UTILS_H
#define UTILS_H
#define FP float

void memcopy_cpu(FP *dst, const FP *src, int size);
void memcopy_gpu(FP *dst, const FP *src, int size);
void matrixmult_cpu(const FP *a, const FP *b, FP *c, int n, int m, int p);
void matrixmult_atrans_cpu(const FP *a, const FP *b, FP *c, int n, int m, int p);
void matrixmult_btrans_cpu(const FP *a, const FP *b, FP *c, int n, int m, int p);
void add_bias_cpu(FP* data, const FP* bias, int size, int channels);
void update_params_cpu(FP* param, const FP* diff, int size, int num, FP lr);

void matrixmult_gpu(bool aTrans, bool bTrans, const FP *a, const FP *b, FP *c, int n, int m, int p,
                                       FP* dev_a, FP* dev_b, FP* dev_c);

void im2col_cpu(const FP* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    FP* data_col);

void col2im_cpu(const FP* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    FP* data_im);

void im2col_gpu(const int channels, const int height, const int width, const int kernel_h,
                const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const FP* data_im, FP* dev_data_im, FP* data_col,
                FP* dev_data_col, const int data_im_size, const int data_col_size,
                const int height_col, const int width_col,
                const int num_kernels, const int num_blocks);

void col2im_gpu(const int channels, const int height, const int width, const int kernel_h,
                const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const FP* data_col, FP* dev_data_col, FP* data_im,
                FP* dev_data_im, const int data_im_size, const int data_col_size,
                const int height_col, const int width_col,
                const int num_kernels, const int num_blocks);

#endif // UTILS_H
