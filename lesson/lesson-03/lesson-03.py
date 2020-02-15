# -*- coding:utf-8 -*-
"""
@file name  : lesson-09-03.py
@author     : tingsongyu
@date       : 2018-08-26
@brief      : 张量操作
"""

import torch
torch.manual_seed(1)

# ======================================= example 1 =======================================
# torch.cat,用于张量的拼接，需要指定拼接的维度

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))

    t_0 = torch.cat([t, t], dim=0)      # shape（4，3）
    t_1 = torch.cat([t, t, t], dim=1)   # shape (2, 9)

    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))


# ======================================= example 2 =======================================
# torch.stack，用于张量的拼接，与cat不同的是，他会创建一个新的维度进行拼接

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))

    t_stack = torch.stack([t, t, t], dim=0)     # shape(3, 2, 3)

    print("\nt_stack:{} shape:{}".format(t_stack, t_stack.shape))


# ======================================= example 3 =======================================
# torch.chunk,用于张量的切分，需要指定切分的维度，若无法整除，
# 则最后一个维度最小

# flag = True
flag = False

if flag:
    a = torch.ones((2, 7))  # 7
    # 三个张量维度分别为（2, 3),（2, 3）, (2, 1)
    list_of_tensors = torch.chunk(a, dim=1, chunks=3)

    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))


# ======================================= example 4 =======================================
# torch.split,张量切分，可以指定每一份大小

# flag = True
flag = False

if flag:
    t = torch.ones((2, 5))
    # 若总和不为5，则会报错
    # list_of_tensors = torch.split(t, [2, 1, 2], dim=1)  # [2 , 1, 2]
    # for idx, t in enumerate(list_of_tensors):
    #     print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))

    list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx, t, t.shape))


# ======================================= example 5 =======================================
# torch.index_select，用于选择指定行或列

# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    # 注意选择张量的type必须是long,且只能是一维
    idx = torch.tensor([0, 2], dtype=torch.long)
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))

# ======================================= example 6 =======================================
# torch.masked_select,用于选取张量中大于或小于某个数的元素，并排成一维张量
# 四种选择函数为le, lt, ge, gt

# flag = True
flag = False

if flag:

    t = torch.randint(0, 9, size=(3, 3))
    mask = t.le(5)  # ge means greater than or equal/   gt: greater than  le  lt
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))


# ======================================= example 7 =======================================
# torch.reshape，改变张量的维度，且更改前后两个张量共享内存地址

# flag = True
flag = False

if flag:
    t = torch.randperm(8)
    t_reshape = torch.reshape(t, (-1, 2, 2))    # -1表示根据其他维度去推断
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))

    t[0] = 1024
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
    print("t.data 内存地址:{}".format(id(t.data)))
    print("t_reshape.data 内存地址:{}".format(id(t_reshape.data)))


# ======================================= example 8 =======================================
# torch.transpose，用于交换张量两个维度的顺序

# flag = True
flag = False

if flag:
    t = torch.rand((2, 3, 4))
    t_transpose = torch.transpose(t, dim0=1, dim1=2)    # c*h*w     h*w*c
    print("t shape:{}\nt_transpose shape: {}".format(t.shape, t_transpose.shape))


# ======================================= example 9 =======================================
# torch.squeeze，用于去除张量中shape为1的维度，若指定维度shape不为1，则不去除

# flag = True
flag = False

if flag:
    t = torch.rand((1, 2, 3, 1))
    t_sq = torch.squeeze(t)
    t_0 = torch.squeeze(t, dim=0)
    t_1 = torch.squeeze(t, dim=1)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)


# ======================================= example 8 =======================================
# torch.add,张量的加法运算

# flag = True
flag = False

if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)  # 运算规则为t_0 + 10*t_1
    t_inplace = t_0.add_(t_1)   # 带_的为in place操作，即直接改变t_0

    print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))














