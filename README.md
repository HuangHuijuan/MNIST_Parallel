# MNIST_Parallel
  This project is implemented based on c++ and cuda. It aims to parallel the computation in CNN to speed up the training process.

  In order to run the program, first, you need to complile files using the provided Makefile, and load the cuda module.
  
  To train a fc network on cpu: ./mnist_fc 0
  
  To train a fc network on gpu: ./mnist_fc 1
  
  To train a cnn network on cpu: ./mnist_cnn 0
  
  To train a cnn network on gpu: ./mnist_cnn 1 
