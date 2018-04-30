'''
[Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
这个文件作为一个可以复用的模块
该模块中自定义 CRF 层，实现批量训练,
构建 CRF 类时需要指定 tag2idx: 字典 以及 ctx，即程序运行在CPU还是 GPU上：默认为 mx.gpu()
调用neg_log_likelihood()函数需要提供两个参数：
    1. feats: 长度为句子长度的列表，列表中每个元素为一个 nd.array，代表一批中每个词的特征向量，每个元素形状为: (batch_size, tagset_size)
    2. tags: 长度为句子长度的列表， 列表中每个元素为一个 nd.array， 代表一批中每个词的标注的索引， 每个元素形状为：(batch_size, 1)

调用 forward()函数进行预测需要提高一个函数:
    1. feats: 同上
        
@author：kenjewu
@date：2018/4/29
'''
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd

ctx = mx.gpu()
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def log_sum_exp(vec):
    # max_score 形状 （batch_size, 1)
    max_score = nd.max(vec, axis=1).reshape((-1, 1))
    # 返回的评分 形状 （batch_size, 1)
    return nd.log(nd.sum(nd.exp(vec - max_score), axis=1)).reshape((-1, 1)) + max_score


class CRF(gluon.nn.Block):
    '''
    自定义 CRF 层
    '''

    def __init__(self, tag2idx, ctx=mx.gpu(), ** kwargs):
        super(CRF, self).__init__(** kwargs)
        with self.name_scope():
            self.tag2idx = tag2idx
            self.ctx = ctx
            self.tagset_size = len(tag2idx)
            # 定义转移评分矩阵的参数, 矩阵中的每个数值代表从状态 j 转移到状态 i 的评分
            self.transitions = self.params.get(
                'transitions', shape=(self.tagset_size, self.tagset_size),
                init=mx.init.Xavier(magnitude=2.24))
            # self.transitions = nd.random.normal(shape=(self.tagset_size, self.tagset_size), ctx=ctx)

    def _forward_alg(self, feats):
        '''
        CRF 概率计算的前向算法
        feats:长度为句子长度的列表，列表中每个元素为一个 nd.array，代表一批中每个词的特征向量，形状为: (batch_size, tagset_size)
        '''
        # 定义前向向量
        batch_size = feats[0].shape[0]
        alphas = [[-10000.] * self.tagset_size]
        alphas[0][self.tag2idx[START_TAG]] = 0.
        alphas = nd.array(alphas, ctx=self.ctx)
        alphas = nd.broadcast_axis(alphas, axis=0, size=batch_size)

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[:, next_tag].reshape((batch_size, -1))
                # trans_score 中的每个分值是从 i 转移到 next_tag 的评分
                trans_score = nd.broadcast_axis(
                    self.transitions.data()[next_tag].reshape((1, -1)),
                    axis=0, size=batch_size)
                next_tag_var = alphas + emit_score + trans_score

                # log_sum_exp(next_tag_var)得到的值的形状： (batch_size, 1)
                alphas_t.append(log_sum_exp(next_tag_var))

            alphas = nd.concat(* alphas_t, dim=1)
        terminal_var = alphas + self.transitions.data()[self.tag2idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        alpha = alpha.reshape((-1, ))
        assert alpha.shape == (batch_size, )
        return alpha

    def _score_sentence(self, feats, tags):
        '''
        计算标注序列的评分
        feats:长度为句子长度的列表，列表中每个元素为一个 nd.array，代表一批中每个词的特征向量，形状为: (batch_size, tagset_size)
        tags: 长度为句子长度的列表， 列表中每个元素为一个 nd.array， 代表一批中每个词的标注的索引， 形状为：(batch_size, 1)
        '''
        batch_size = feats[0].shape[0]
        score = nd.ones((batch_size,), ctx=self.ctx)

        # 检索一批句子符号序列的开始标签的矩阵， 形状为：(batch_size, 1)
        temp = nd.array([self.tag2idx[START_TAG]] * batch_size, ctx=self.ctx).reshape((batch_size, 1))
        # 拼接, 结果形状为： (batch_size, max_seq_len + 1)
        tags = nd.concat(temp, *tags, dim=1)

        for i, feat in enumerate(feats):
            score = score + nd.pick(self.transitions.data()[tags[:, i + 1]], tags[:, i], axis=1) + \
                nd.pick(feat, tags[:, i + 1], axis=1)
        score = score + self.transitions.data()[self.tag2idx[STOP_TAG], tags[:, tags.shape[1]-1]]
        return score

    def _viterbi_decode(self, feats):
        '''
        CRF 的预测算法，维特比算法,即根据特征找出最好的路径
        feats:长度为句子长度的列表，列表中每个元素为一个 nd.array，代表一批中每个词的特征向量，形状为: (batch_size, tagset_size)
        '''
        backpointers = []
        batch_size = feats[0].shape[0]
        vvars = nd.full((1, self.tagset_size), -10000., ctx=self.ctx)
        vvars[0, self.tag2idx[START_TAG]] = 0
        # vvars 形状：(batch_size, tagset_size)
        vvars = nd.broadcast_axis(vvars, axis=0, size=batch_size)

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = vvars + nd.broadcast_axis(
                    self.transitions.data()[next_tag].reshape((1, -1)),
                    axis=0, size=batch_size)
                # best_tag_id 形状（batch_size, 1)
                best_tag_id = nd.argmax(next_tag_var, axis=1, keepdims=True)
                bptrs_t.append(best_tag_id)
                # viterbivars_t 列表中每个元素的形状为 (batch_size, 1)
                viterbivars_t.append(nd.pick(next_tag_var, best_tag_id, axis=1, keepdims=True))
            vvars = (nd.concat(* viterbivars_t, dim=1) + feat)
            # bptrs_t 形状 ：(batch_size, tagset_size)
            bptrs_t = nd.concat(*bptrs_t, dim=1)
            backpointers.append(bptrs_t)

        # 转换到 STOP_TAG
        terminal_var = vvars + self.transitions.data()[self.tag2idx[START_TAG]]
        best_tag_id = nd.argmax(terminal_var, axis=1)
        # path_score 形状（batch_size, )
        path_score = nd.pick(terminal_var, best_tag_id, axis=1)

        # 根据反向指针 backpointers 去解码最好的路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = nd.pick(bptrs_t, best_tag_id, axis=1)
            best_path.append(best_tag_id)
        # 移除开始符号
        # start 形状为 (batch_size, )
        start = best_path.pop()
        # 检查start是否为开始符号
        for i in range(batch_size):
            assert start[i].asscalar() == self.tag2idx[START_TAG]
        best_path.reverse()

        # 构建最佳路径的矩阵
        new_best_path = []
        for best_tag_id in best_path:
            best_tag_id = best_tag_id.reshape((-1, 1))
            new_best_path.append(best_tag_id)
        best_path_matrix = nd.concat(*new_best_path, dim=1)

        return path_score, best_path_matrix

    def neg_log_likelihood(self, feats, tags):
        '''
        计算CRF中标签序列的对数似然
        feats: 长度为句子长度的列表，列表中每个元素为一个 nd.array，代表一批中每个词的特征向量，形状为: (batch_size, tagset_size)
        tags: 长度为句子长度的列表， 列表中每个元素为一个 nd.array， 代表一批中每个词的标注的索引， 形状为：(batch_size, 1)
        '''
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        # 返回值形状： (batch_size, )
        return forward_score - gold_score

    def forward(self, feats):
        '''
        根据特征进行预测，可以批量化操作
        feats: 长度为句子长度的列表，列表中每个元素为一个 nd.array，代表一批中每个词的特征向量，形状为: (batch_size, tagset_size)

        '''
        score, tag_seq = self._viterbi_decode(feats)
        # 返回 score 形状：（batch_size,), tag_seq 形状:(batch_size, max_seq_len)
        return score, tag_seq
