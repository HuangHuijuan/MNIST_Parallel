TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    utils.cpp \
    relulayer.cpp \
    softmaxlayer.cpp \
    softmaxlosslayer.cpp \
    mnistdatalayer.cpp \
    test_relu.cpp \
    test_softmax.cpp \
    test_datalayer.cpp

HEADERS += \
    convlayer.h \
    innerproductlayer.h \
    utils.h \
    relulayer.h \
    softmaxlayer.h \
    softmaxlosslayer.h \
    mnistdatalayer.h

DISTFILES += \
    utils.cu \
    test_conv.cu \
    test_fc.cu \
    mnist_fc.cu \
    mnist_cnn.cu \
    innerproductlayer.cu \
    convlayer.cu \
    test_im2col.cu
