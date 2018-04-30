# CRF
Linear chain conditional random fields are implemented using numpy and mxnet. 
The code has been modified with reference to pytroch's crf tutorial. 
Achieve batch training instead of training with only one item of data. 
CRF is designed as a layer of neural networks that can be reused in mxet/gluon.
If you do not use mxnet/gluon, you can download the crf.py file and modify the code to suit your application.
* [crf.py](./crf.py)
* [test.py](./test.py)
* [test_crf_layer](./test_crf_layer.py)
***
使用 numpy 和 mxnet 实现了线性链条件随机场。代码参照 pytroch 的 crf 教程进行了修改。
实现了批量训练，而不再是只能一条一条数据的训练。并将 CRF 设计为在mxet/gluon中可以复用的神经网络的层。
如果你不使用mxnet/gluon，可以下载crf.py 文件将代码进行修改，以适用于你的程序。

