#include <cuda.h>
#include <math.h>
#include <iostream>
#include "utils.h"
#define Block_Dim 32
#define NUM_THREADS Block_Dim * Block_Dim
using namespace std;

void memcopy_gpu(FP *dst, const FP *src, int size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
}
__global__ void matmul_tiled_kernel(const FP *a, const FP *b, FP *c, int n, int m, int p) {
  int TW = blockDim.x;
  extern __shared__ FP bigarray[];
  FP *atile = &bigarray[0], *btile = &bigarray[TW * TW];
  int t, k, i, j, aIndex, bIndex, atIndex, btIndex;
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = tx + blockDim.x * blockIdx.x;
  int row = ty + blockDim.y * blockIdx.y;

  FP cvalue = 0.;
  aIndex = row * p + tx;
  atIndex = ty * TW + tx;
  bIndex = ty * m + col;
  btIndex = ty * TW + tx;
  for (t = 0; t * TW < p; t++, aIndex += TW, bIndex += TW * m) {
    if (row < n && t * TW + tx < p) {
      atile[atIndex] = a[aIndex];
    }
    if (col < m && t * TW + ty < p) {
      btile[btIndex] = b[bIndex];
    }
    __syncthreads();
    if(col < m && row < n) {
      for (k = 0, i = ty * TW, j = tx; k < min(p - t * TW, TW); k++, i++, j+=TW)
        cvalue += atile[i] * btile[j];
    }
    __syncthreads();
  }
  if(col < m && row < n) {
    c[row * m + col] = cvalue;
  }
}

__global__ void atrans_matmul_tiled_kernel(const FP *a, const FP *b, FP *c, int n, int m, int p) {

  int TW = blockDim.x;

  extern __shared__ FP bigarray[];
  FP *atile = &bigarray[0], *btile = &bigarray[TW * TW];
  int t, k, i, j, aIndex, bIndex, atIndex, btIndex;
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = tx + blockDim.x * blockIdx.x;
  int row = ty + blockDim.y * blockIdx.y;

  FP cvalue = 0.;
  aIndex = tx * n + row;
  atIndex = ty * TW + tx;
  bIndex = ty * m + col;
  btIndex = ty * TW + tx;
  for (t = 0; t * TW < p; t++, aIndex += TW * n, bIndex += TW * m) {
    if (row < n && t * TW + tx < p) {
      atile[atIndex] = a[aIndex];
    }
    if (col < m && t * TW + ty < p) {
      btile[btIndex] = b[bIndex];
    }
    __syncthreads();
    if(col < m && row < n) {
      for (k = 0, i = ty * TW, j = tx; k < min(p - t * TW, TW); k++, i++, j+=TW)
        cvalue += atile[i] * btile[j];
    }
    __syncthreads();
  }
  if(col < m && row < n) {
    c[row * m + col] = cvalue;
  }
}

__global__ void btrans_matmul_tiled_kernel(const FP *a, const FP *b, FP *c, int n, int m, int p) {

    int TW = blockDim.x;

    extern __shared__ FP bigarray[];
    FP *atile = &bigarray[0], *btile = &bigarray[TW * TW];
    int t, k, i, j, aIndex, bIndex, atIndex, btIndex;
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = tx + blockDim.x * blockIdx.x;
    int row = ty + blockDim.y * blockIdx.y;

    FP cvalue = 0.;
    aIndex = row * p + tx;
    atIndex = ty * TW + tx;
    bIndex = col * p + ty;
    btIndex = ty * TW + tx;
    for (t = 0; t * TW < p; t++, aIndex += TW, bIndex += TW) {
      if (row < n && t * TW + tx < p) {
        atile[atIndex] = a[aIndex];
      }
      if (col < m && t * TW + ty < p) {
        btile[btIndex] = b[bIndex];
      }
      __syncthreads();
      if(col < m && row < n) {
        for (k = 0, i = ty * TW, j = tx; k < min(p - t * TW, TW); k++, i++, j+=TW)
          cvalue += atile[i] * btile[j];
      }
      __syncthreads();
    }
    if(col < m && row < n) {
      c[row * m + col] = cvalue;
    }
}

void matrixmult_gpu(bool aTrans, bool bTrans, const FP *a, const FP *b, FP *c, int n, int m, int p,
                    FP* dev_a, FP* dev_b, FP* dev_c) {

    cudaMemcpy(dev_a, a , n * p * sizeof(FP) ,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b , p * m * sizeof(FP),cudaMemcpyHostToDevice);

    int Grid_Dim = ceil(max(n, m) / float(Block_Dim));
    dim3 Grid(Grid_Dim, Grid_Dim); //Grid structure
    dim3 Block(Block_Dim, Block_Dim); //Block structure

    //size of dynamic shared memory
    int Ns = 2 * Block_Dim * Block_Dim * sizeof(FP);

    if (aTrans && !bTrans) {
        atrans_matmul_tiled_kernel<<<Grid, Block, Ns>>>(dev_a, dev_b, dev_c, n, m, p);
    } else if (!aTrans && bTrans) {
        btrans_matmul_tiled_kernel<<<Grid, Block, Ns>>>(dev_a, dev_b, dev_c, n, m, p);
    } else if (!aTrans && !bTrans) {
        matmul_tiled_kernel<<<Grid, Block, Ns>>>(dev_a, dev_b, dev_c, n, m, p);
    }

    cudaMemcpy(c,dev_c, n * m * sizeof(FP),cudaMemcpyDeviceToHost);

}

__global__ void im2col_gpu_kernel(const int n, const FP* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    FP* data_col) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        FP* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const FP* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
          for (int j = 0; j < kernel_w; ++j) {
            int h_im = h_offset + i;
            int w_im = w_offset + j;
            *data_col_ptr =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                data_im_ptr[i * width + j] : 0;
            data_col_ptr += height_col * width_col;
          }
        }
  }
}

void im2col_gpu(const int channels, const int height, const int width, const int kernel_h,
                const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const FP* data_im, FP* dev_data_im, FP* data_col,
                FP* dev_data_col, const int data_im_size, const int data_col_size,
                const int height_col, const int width_col,
                const int num_kernels, const int num_blocks) {
      // We are going to launch channels * height_col * width_col kernels, each
      // kernel responsible for copying a single-channel grid.

      cudaMemcpy(dev_data_im, data_im, data_im_size ,cudaMemcpyHostToDevice);

      im2col_gpu_kernel<<<num_blocks, NUM_THREADS>>>(
          num_kernels, dev_data_im, height, width, kernel_h, kernel_w, pad_h,
          pad_w, stride_h, stride_w, height_col, width_col, dev_data_col);

      cudaMemcpy(data_col, dev_data_col, data_col_size, cudaMemcpyDeviceToHost);

}

__global__ void col2im_gpu_kernel(const int n, const FP* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    FP* data_im) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
        FP val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);
        // compute the start and end of the output
        const int w_col_start =
            (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);

        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
          for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
            int h_k = (h_im - h_col * stride_h);
            int w_k = (w_im - w_col * stride_w);
            int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                    height_col + h_col) * width_col + w_col;
              val += data_col[data_col_index];
          }
        }
        data_im[index] = val;
  }
}

void col2im_gpu(const int channels, const int height, const int width, const int kernel_h,
                const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const FP* data_col, FP* dev_data_col, FP* data_im,
                FP* dev_data_im, const int data_im_size, const int data_col_size,
                const int height_col, const int width_col,
                const int num_kernels, const int num_blocks) {

      cudaMemcpy(dev_data_col, data_col , data_col_size ,cudaMemcpyHostToDevice);
      col2im_gpu_kernel<<<num_blocks, NUM_THREADS>>>(
          num_kernels, dev_data_col, height, width, channels, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, height_col, width_col, dev_data_im);

      cudaMemcpy(data_im, dev_data_im, data_im_size, cudaMemcpyDeviceToHost);
}


