#include <iostream>
#include <memory.h>
#include "utils.h"
using namespace std;

void memcopy_cpu(FP *dst, const FP *src, int size) {
    memcpy(dst, src, size);
}
void matrixmult_cpu(const FP *a, const FP *b, FP *c, int n, int m, int p) {
    int i, j, k, index, indexb;
    FP r;
    for (k = 0; k < p; k++) {
        for (i = 0; i < n; i++) {
            r = a[i * p + k];
            index = i * m;
            indexb = k * m;
            for (j = 0; j < m; j++) {
                c[index + j] += r * b[indexb + j];
            }
        }
    }
}

void matrixmult_atrans_cpu(const FP *a, const FP *b, FP *c, int n, int m, int p) {
  int i, j, k, index, indexb;
  FP r;
  for (k = 0; k < p; k++) {
    for (i = 0; i < n; i++) {
      r = a[k * n + i];
      index = i * m;
      indexb = k * m;
      for (j = 0; j < m; j++)
        c[index + j] += r * b[indexb + j];
    }
  }
}

void matrixmult_btrans_cpu(const FP *a, const FP *b, FP *c, int n, int m, int p) {
    int i, j, k, index, indexb;
    FP r;
    for (k = 0; k < p; k++) {
        for (i = 0; i < n; i++) {
            r = a[i * p + k];
            index = i * m;
            indexb = k;
            for (j = 0; j < m; j++, indexb += p) {
                c[index + j] += r * b[indexb];
            }
        }
    }
}

void add_bias_cpu(FP* data, const FP* bias, int size, int channels) {
    for (int i = 0; i < channels; i++) {
        for (int j = 0; j < size; j++) {
            data[i * size + j] += bias[i];
        }
    }
}

void update_params_cpu(FP* param, const FP* diff, int size, int num, FP lr) {
    for (int i = 0; i < size; i++) {
        param[i] -=  lr * diff[i];
    }
}

void im2col_cpu(const FP* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    FP* data_col) {
  const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1; //the height of output feature maps
  const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1; //the weight of output feature maps
  const int channels_col = channels * kernel_h * kernel_w;
  // traverse a kernel size of image
  for (int c = 0; c < channels_col; ++c) {
    // w offset in the current kernel
    int w_offset = c % kernel_w;
    // h offset in the current kernel
    int h_offset = (c / kernel_w) % kernel_h;
   // the cth channel of the current kernel
    int c_im = c / kernel_h / kernel_w;

    // traverse every position of the output feature maps
    for (int h = 0; h < height_out; ++h) {
      for (int w = 0; w < width_out; ++w) {
        // the pixel (h, w) correspond to the region [h * stride_h - pad_h,   h * stride_h - pad_h+kernel_h],
        // [w * stride_w - pad_w,   w * stride_w - pad_w+kernel_w] of the input feature map
        int h_im = h * stride_h - pad_h + h_offset;  //the mapped h position of the input feature map
        int w_im = w * stride_w - pad_w + w_offset;  //the mapped w position of the input feature map
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_col[(c * height_out + h) * width_out + w] =
            data_im[(c_im * height + h_im) * width + w_im];
        else
          data_col[(c * height_out + h) * width_out + w] = 0;
      }
    }
  }
}

void col2im_cpu(const FP* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    FP* data_im) {
  int height_out = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_out = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_out; ++h) {
      for (int w = 0; w < width_out; ++w) {
        int h_im = h * stride_h - pad_h + h_offset;
        int w_im = w * stride_w - pad_w + w_offset;
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
              data_col[(c * height_out + h) * width_out + w];
      }
    }
  }
}
