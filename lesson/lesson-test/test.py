#   _*_coding:utf-8 _*_
#   author:wenbinan
#   time:2019/11/8 21:24
#   filename:test.py
#   product:PyCharm

import numpy as np
import torch
import torchvision
import random


class Animal:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind

    def shout(self):
        print("I am a {0}".format(self.name))


animal = Animal("cat", "pet")
# animal.shout()
# print(animal.name)
print(np.zeros(3).shape)
