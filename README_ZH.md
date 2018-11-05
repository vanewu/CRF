# 使用 Mxnet/Gluon 实现条件随机场
![GitHub license](https://img.shields.io/badge/license-Apache2.0-blue.svg)
[![CRF](https://img.shields.io/badge/Moudle-CRF-green.svg)](./crf.py)
## 介绍
使用 Numpy 和 Mxnet/Gluon 实现了线性链条件随机场，并且支持批量训练，而不局限于将数据一条一条的训练。最新版本代码使用 [foreach](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html?highlight=fore#mxnet.ndarray.contrib.foreach) 操作进行序列的展开计算，计算中尽可能的使用 [NDArray](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html?highlight=nd#module-mxnet.ndarray) 进行计算，以便于在 GPU 上加速计算， 最后测试比之前使用 for 循环展开的计算速度快了40多倍，性能得到有效提高。

[crf.py](./crf.py) 中实现了条件随机场中的前向计算，后向计算，维特比解码，对数极大似然等方法，最后将 CRF 设计为在 Mxet/Gluon中可以复用的神经网络的层。以便于直接与其它网络层对接训练，比如 LSTM + CRF 等。

## 目录

* [crf.py](./crf.py)  (CRF 实现的源代码)
* [test.py](./test.py)  (直接测试 CRF 是否可用)
* [test_crf_layer.py](./test_crf_layer.py)  (测试将CRF作为一个层放入其他神经网络模型是否可用)

## 引用
* [Mxnet](https://mxnet.incubator.apache.org/api/python/index.html)
* [Numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html)
* [统计学习方法(李航)](https://baike.baidu.com/item/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/10430179?fr=aladdin)

