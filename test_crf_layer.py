'''
测试 CRF 模块文件，这里是通过将 CRF 作为一个层放入其他模型中测试，
测试自定义的 CRF 层是否能够复用
@author：kenjewu
@date：2018/4/30
'''
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from crf import CRF

ctx = mx.gpu()
START_TAG = "<START>"
STOP_TAG = "<STOP>"

# 构造假的数据集用于测试 CRF 模型是否能够正常运行
tag2idx = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

x = nd.random.normal(shape=(10, 5), ctx=ctx)
y = nd.array([[0, 1, 0, 2, 1, 0, 1],
              [0, 2, 0, 0, 2, 0, 1],
              [1, 1, 1, 0, 1, 0, 2],
              [0, 0, 2, 2, 0, 1, 0],
              [1, 1, 1, 1, 2, 2, 1],
              [0, 1, 2, 2, 0, 0, 1],
              [2, 2, 0, 2, 0, 1, 1],
              [1, 1, 2, 0, 1, 0, 0],
              [0, 2, 1, 2, 1, 2, 0],
              [0, 1, 2, 0, 1, 1, 2]], ctx=ctx)

dataset_train = gluon.data.ArrayDataset(x, y)
iter_train = gluon.data.DataLoader(dataset_train, batch_size=5, shuffle=True)


class CRF_MODEL(gluon.nn.Block):
    '''
    这里构造一个其他模型，虽然模型中只有 CRF 一层，但可以测试 CRF 是否能作为自定义层复用
    '''

    def __init__(self, tag2idx, ctx=mx.gpu(), ** kwargs):
        super(CRF_MODEL, self).__init__(** kwargs)
        with self.name_scope():
            self.crf = CRF(tag2idx, ctx=ctx)

    def forward(self, x):
        return self.crf(x)


# 构建一个模型
model = CRF_MODEL(tag2idx, ctx=ctx)
model.initialize(ctx=ctx)
# 查看模型中的参数
print(model.collect_params())
print(model.collect_params()['crf_model0_crf0_transitions'].data())
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 1e-4})

# 训练
for epoch in range(100):
    for batch_x, batch_y in iter_train:
        # CRF 的输入要求是一个长度为序列长的列表，所以这里是在构造这个列表，当然这个数据是假的
        batch_x = [batch_x] * 7
        batch_y = nd.split(batch_y, 7, axis=1)
        with autograd.record():
            # 求对数似然
            neg_log_likelihood = model.crf.neg_log_likelihood(batch_x, batch_y)
        # 求导并更新
        neg_log_likelihood.backward()
        optimizer.step(5)
print(model.collect_params()['crf_model0_crf0_transitions'].data())

# 使用模型预测
print(model([x] * 7))
