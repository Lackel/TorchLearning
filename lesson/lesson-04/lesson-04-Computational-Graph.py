# -*- coding:utf-8 -*-
"""
@file name  : lesson-09-04-Computational-Graph.py
@author     : tingsongyu
@date       : 2018-08-28
@brief      : 计算图示例
"""
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)     # retain_grad()
# a.retain_grad()     # 如果想查看非叶子节点的梯度，需要该函数，否则返回None
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
# print(a.grad)    # 不加a.retain_grad(),返回为None
print(w.grad)

# 查看叶子结点
# print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
# print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
# print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)

