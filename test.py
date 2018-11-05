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

# Test directly if the CRF module can be instantiated and used
# @author：kenjewu
# @date：2018/10/05

import time
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, nd
from crf import CRF


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


CTX = try_gpu()
START_TAG = "<eos>"
STOP_TAG = "<bos>"

# generate pseudo data
tag2idx = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

x = nd.random.normal(shape=(10, 5), ctx=CTX)
y = nd.array([[0, 1, 0, 2, 1, 0, 1],
              [0, 2, 0, 0, 2, 0, 1],
              [1, 1, 1, 0, 1, 0, 2],
              [0, 0, 2, 2, 0, 1, 0],
              [1, 1, 1, 1, 2, 2, 1],
              [0, 1, 2, 2, 0, 0, 1],
              [2, 2, 0, 2, 0, 1, 1],
              [1, 1, 2, 0, 1, 0, 0],
              [0, 2, 1, 2, 1, 2, 0],
              [0, 1, 2, 0, 1, 1, 2]], ctx=CTX)

dataset_train = gluon.data.ArrayDataset(x, y)
iter_train = gluon.data.DataLoader(dataset_train, batch_size=5, shuffle=True)
# Build a CRF model directly
model = CRF(tag2idx, ctx=CTX)
model.initialize(ctx=CTX)
# print params of the model
print(model.collect_params())
print(model.collect_params()['crf0_transitions'].data())
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 1e-4})

# train
start_time = time.time()

for epoch in range(100):
    for batch_x, batch_y in iter_train:
        batch_x = nd.broadcast_axis(nd.expand_dims(batch_x, axis=0), axis=0, size=7)
        with autograd.record():
            # loss
            neg_log_likelihood = model.neg_log_likelihood(batch_x, batch_y)
        # backward and update params
        neg_log_likelihood.backward()
        optimizer.step(5)
print(model.collect_params()['crf0_transitions'].data())
print(model(nd.broadcast_axis(nd.expand_dims(x, axis=0), axis=0, size=7)))

print('use {0} secs!'.format(time.time()-start_time))
