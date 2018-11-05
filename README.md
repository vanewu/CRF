# Conditional Random Field With Mxnet/Gluon
![GitHub license](https://img.shields.io/badge/license-Apache2.0-blue.svg)
[![CRF](https://img.shields.io/badge/Moudle-CRF-green.svg)](./crf.py)
## Introduction
Linear chain conditional random fields are implemented using Numpy and Mxnet/Gluon, and batch training is supported, not limited to training the data one by one. The latest version of the code uses the [foreach](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html?highlight=fore#mxnet.ndarray.contrib.foreach)  operation to perform the expansion of the sequence. The calculation uses [NDArray](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html?highlight=nd#module-mxnet.ndarray)  for calculations as much as possible, so that the calculation can be accelerated on the GPU. The final test is about 6 times faster than the previous calculation using the for loop.

The forward calculation, backward calculation, Viterbi decoding, log maximum likelihood and other methods in the conditional random field are realized in [crf.py](./crf.py). Finally, the CRF is designed as the layer of the neural network that can be reused in Mxet/Gluon. In order to directly interface with other network layers, such as LSTM + CRF.

## Contents
* [crf.py](./crf.py)  (Source code for CRF implementation)
* [test.py](./test.py)  (Directly test whether CRF is available)
* [test_crf_layer.py](./test_crf_layer.py)  (Place the CRF as a custom layer in other models and test for availability)

## Reference
* [MXNet - Python API](https://mxnet.incubator.apache.org/api/python/index.html)
* [Numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html)
* [统计学习方法(李航)](https://baike.baidu.com/item/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/10430179?fr=aladdin)

