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


def log_sum_exp(vec):
    # max_score 形状 （self.tagset_size, batch_size, 1)
    max_score = nd.max(vec, axis=-1, keepdims=True)
    # score 形状 （self.tagset_size, batch_size, 1)
    score = nd.log(nd.sum(nd.exp(vec - max_score), axis=-1, keepdims=True)) + max_score

    # 返回的 NDArray shape: (self.tagset_size, batch_size, )
    return nd.squeeze(score, axis=-1)


class CRF(gluon.nn.Block):
    '''
    自定义 CRF 层
    '''

    def __init__(self, tag2idx, ctx=CTX, ** kwargs):
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
        '''CRF 概率计算的前向算法

        Args:
            feats (NDArray): 代表一批中每个词的特征向量，形状为: (seq_len, batch_size, tagset_size)

        Returns:
            [type]: [description]
        '''

        # 定义前向向量
        batch_size = feats.shape[1]

        # alphas shape: [batch_size, self.tagset_size]
        alphas = nd.full((batch_size, self.tagset_size), -10000., ctx=self.ctx)
        alphas[:, self.tag2idx[START_TAG]] = 0.

        def update_alphas(data, alphas):
            '''计算每个时间步批量更新 alpha

            Args:
                data (NDArray): NDArray shape: [seq_len, batch_size, self.tagset_size]
                alphas (NDArray): NDArray shape: [batch_size, self.tagset_size]
            '''

            # alphas_t shape: [self.tagset_size, batch_size, self.tagset_size]
            alphas_t = nd.broadcast_axis(nd.expand_dims(alphas, axis=0), axis=0, size=self.tagset_size)
            # emit_score shape: [self.tagset_size, batch_size, 1]
            emit_score = nd.transpose(nd.expand_dims(data, axis=0), axes=(2, 1, 0))
            # trans_score shape: [self.tagset_size, 1, self.tagset_size]
            trans_score = nd.expand_dims(self.transitions.data(), axis=1)

            # next_tag_var shape: [self.tagset_size, batch_size, self.tagset_size]
            next_tag_var = alphas_t + emit_score + trans_score

            # alphas shape: [self.tagset_size, batch_size]
            alphas = log_sum_exp(next_tag_var)
            # alphas shape: [batch_size, self.tagset_size]
            alphas = nd.transpose(alphas, axes=(1, 0))

            return data, alphas

        _, alphas = nd.contrib.foreach(update_alphas, feats, alphas)

        # terminal_var shape:[batch_size, self.tagset_size]
        terminal_var = alphas + self.transitions.data()[self.tag2idx[STOP_TAG]]
        # alpha shape: [batch_size, ]
        alpha = log_sum_exp(terminal_var)
        assert alpha.shape == (batch_size, )
        return alpha

    def _score_sentence(self, feats, tags):
        '''计算标注序列的评分

        Args:
            feats (NDArray): 代表一批中每个词的特征向量，形状为: (seq_len, batch_size, self.tagset_size)
            tags (NDArray): 代表一批中每个词的标注的索引， 形状为：(batch_size, seq_len)

        Returns:
            score (NDArray): shape: (batch_size, )
        '''

        batch_size = feats.shape[1]
        score = nd.zeros((batch_size,), ctx=self.ctx)

        # 检索一批句子符号序列的开始标签的矩阵， 形状为：(batch_size, 1)
        temp = nd.array([self.tag2idx[START_TAG]] * batch_size,
                        ctx=self.ctx).reshape((batch_size, 1))
        # 拼接, 结果形状为： (batch_size, seq_len+1)
        tags = nd.concat(temp, tags, dim=1)

        def update_score(data, states):
            '''计算评分

            Args:
                data (NDArray): NDArray shape:(seq_len, batch_size, self.tagset_size)
                states (list of NDArray): [idx, tags, score]

            Returns:
                score (NDArray): NDarray shape: (batch_size,)
            '''
            # feat shape: (batch_size, self.tagset_size)
            feat = data
            # tag shape:(batch_size, 1)
            idx, tags, score = states
            i = int(idx.asscalar())
            score = score + nd.pick(self.transitions.data()[tags[:, i + 1]],
                                    tags[:, i], axis=1) + nd.pick(feat, tags[:, i + 1], axis=1)
            idx += 1

            return feat, [idx, tags, score]

        states = [nd.array([0]), tags, score]
        _, states = nd.contrib.foreach(update_score, feats, states)
        score = states[2]
        score = score + self.transitions.data()[self.tag2idx[STOP_TAG], tags[:, int(tags.shape[1]-1)]]
        return score

    def _viterbi_decode(self, feats):
        '''
        CRF 的预测算法，维特比算法,即根据特征找出最好的路径

        Args:
            feats (NDArray): 代表一批中每个词的特征向量，形状为: (seq_len, batch_size, self.tagset_size)

        Returns:
            [type]: [description]
        '''

        backpointers = []
        batch_size = feats.shape[1]

        # vvars shape：(batch_size, self.tagset_size)
        vvars = nd.full((batch_size, self.tagset_size), -10000., ctx=self.ctx)
        vvars[:, self.tag2idx[START_TAG]] = 0.0

        def update_decode(data, states):
            feat = data
            vvars = states

            # vvars_t shape: [self.tagset_size, batch_size, self.tagset_size]
            vvars_t = nd.broadcast_axis(nd.expand_dims(vvars, axis=0), axis=0, size=self.tagset_size)
            # trans shape: [self.tagset_size, 1, self.tagset_size]
            trans = nd.expand_dims(self.transitions.data(), axis=1)
            next_tag_var = vvars_t + trans

            # best_tag_id shape: [self.tagset_size, batch_size]
            best_tag_id = nd.argmax(next_tag_var, axis=-1)

            # bptrs_t, viterbivars_t  shape ：(batch_size, tagset_size)
            viterbivars_t = nd.transpose(nd.pick(next_tag_var, best_tag_id, axis=-1), axes=(1, 0))
            bptrs_t = nd.transpose(best_tag_id, axes=(1, 0))

            vvars = viterbivars_t + feat

            return bptrs_t, vvars

        # backpointers shape: [seq_len, batch_size, self.tagset_size]
        backpointers, vvars = nd.contrib.foreach(update_decode, feats, vvars)

        # 转换到 STOP_TAG
        terminal_var = vvars + self.transitions.data()[self.tag2idx[START_TAG]]
        best_tag_id = nd.argmax(terminal_var, axis=1)
        # path_score 形状（batch_size, )
        path_score = nd.pick(terminal_var, best_tag_id, axis=1)

        # 根据反向指针 backpointers 去解码最好的路径

        best_path = [best_tag_id]
        for bptrs_t in nd.reverse(backpointers, axis=0):
            # best_tag_id shape: (batch_size, )
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
        best_path_matrix = nd.stack(*best_path, axis=0)
        return path_score, best_path_matrix

    def neg_log_likelihood(self, feats, tags):
        '''计算CRF中标签序列的对数似然

        Args:
            feats (NDArray): 代表一批中每个词的特征向量，形状为: (seq_len, batch_size, self.tagset_size)
            tags (NDArray): 代表一批中每个词的标注的索引， 形状为：(batch_size, seq_len)

        Returns:
            [NDArray]: shape: (batch_size, )
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
