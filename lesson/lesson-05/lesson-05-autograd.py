# -*- coding: utf-8 -*-
"""
# @file name  : lesson-09-05-autograd.py
# @author     : tingsongyu
# @date       : 2019-08-30 10:08:00
# @brief      : torch.autograd
"""
import torch
torch.manual_seed(10)


# ====================================== retain_graph ==============================================
flag = True
# flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)  # 若没有retain_graph,则无法进行第二次反向传播
    print(w.grad)   # w的梯度为5
    y.backward()  # 第二次反向传播，由于没有清零梯度，w的梯度变为10
    # w.grad.zero_()    # 运行该语句后w的梯度为5
    print(w.grad)

# ====================================== grad_tensors ==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)    dy0/dw = x+2w+1 = 5
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)       # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])

    # 传入的参数为loss中两个梯度要乘的系数
    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad)   # 结果为9，计算方法为 1*5 + 2*2


# ====================================== autograd.gard ==============================================
# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)     # y = x**2

    # 一次求导
    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
    print(grad_1)   # grad_1为一个tuple

    # 二次求导
    grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)


# ====================================== tips: 1 ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        w.grad.zero_()      # 若没有该语句，梯度会反复叠加，即每次乘2


# ====================================== tips: 2 ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # 根据叶子节点创建的张量，requires_grad自动设置为True
    print(a.requires_grad, b.requires_grad, y.requires_grad)


# ====================================== tips: 3 ==============================================
# flag = True
flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)

    # 用 a = a+ 操作，a的地址会发生改变
    a = a + torch.ones((1, ))
    print(id(a), a)

    # 用a+=操作，a的地址不变，相当于原位操作
    # a += torch.ones((1, ))
    # print(id(a), a)


# ====================================== tips: 4 ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # 不能对requires_grad=True叶子节点进行原位操作，即改变叶子节点的值，否则会报错
    w.add_(1)

    y.backward()





