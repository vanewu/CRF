# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Use CRF as a neural network layer built by GLuon to conduct training and prediction tests.
# @author：kenjewu
# @date：2018/10/05

import time
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from crf import CRF

ctx = mx.gpu()
START_TAG = "<bos>"
STOP_TAG = "<eos>"

# generate pseudo data
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
    '''Here we construct a neural network.
       Although there is only one CRF layer in the model,
       we can test whether CRF can be reused as a custom layer.

    Args:
        gluon ([type]): [description]

    Returns:
        [type]: [description]
    '''

    def __init__(self, tag2idx, ctx=mx.gpu(), ** kwargs):
        super(CRF_MODEL, self).__init__(** kwargs)
        with self.name_scope():
            self.crf = CRF(tag2idx, ctx=ctx)

    def forward(self, x):
        return self.crf(x)


# build a model
model = CRF_MODEL(tag2idx, ctx=ctx)
model.initialize(ctx=ctx)
# print params of the model
print(model.collect_params())
print(model.collect_params()['crf_model0_crf0_transitions'].data())
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 1e-4})

# train
start_time = time.time()
for epoch in range(100):
    for batch_x, batch_y in iter_train:
        batch_x = nd.broadcast_axis(nd.expand_dims(batch_x, axis=0), axis=0, size=7)
        with autograd.record():
            # loss
            neg_log_likelihood = model.crf.neg_log_likelihood(batch_x, batch_y)
        # backward and update params
        neg_log_likelihood.backward()
        optimizer.step(5)
print(model.collect_params()['crf_model0_crf0_transitions'].data())

# predict
print(model(nd.broadcast_axis(nd.expand_dims(x, axis=0), axis=0, size=7)))

print('use {0} secs!'.format(time.time()-start_time))
