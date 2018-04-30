# CRF
Linear chain conditional random fields are implemented using numpy and mxnet. 
The code has been modified with reference to pytroch's crf tutorial. 
Achieve batch training instead of training with only one item of data. 
CRF is designed as a layer of neural networks that can be reused in mxet/gluon.
If you do not use mxnet/gluon, you can download the crf.py file and modify the code to suit your application.
* [crf.py](./crf.py)  (Source code for CRF implementation)
* [test.py](./test.py)  (Directly test whether CRF is available)
* [test_crf_layer](./test_crf_layer.py)  (Place the CRF as a custom layer in other models and test for availability)

# Reference
* [Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
* [统计学习方法(李航)](https://baike.baidu.com/item/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/10430179?fr=aladdin)
* [MXNet - Python API](https://mxnet.incubator.apache.org/api/python/index.html)
* [Numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html)
***
使用 numpy 和 mxnet 实现了线性链条件随机场。代码参照 pytroch 的 crf 教程进行了修改。
实现了批量训练，而不再是只能一条一条数据的训练。并将 CRF 设计为在mxet/gluon中可以复用的神经网络的层。
如果你不使用mxnet/gluon，可以下载crf.py 文件将代码进行修改，以适用于你的程序。
* [crf.py](./crf.py)  (CRF 实现的源代码)
* [test.py](./test.py)  (直接测试 CRF 是否可用)
* [test_crf_layer](./test_crf_layer.py)  (测试将CRF作为一个层放入其他神经网络模型是否可用)
