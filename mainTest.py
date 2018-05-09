# -*- coding: utf-8 -*-
import tree
import copy
dataset,label = tree.createDataSet()
print(label)
# 这里仅仅用 labels=label是不行的，因为它们指向同一个内存
labels=copy.deepcopy(label)
myTree = tree.createTree(dataset,labels)
# print(myTree)
print(label)
testResult = tree.classify(myTree,label,[1,1])
print(testResult)
tree.storeTree(myTree,"F:\NatureRecognition/tree.txt")
tt=tree.grabTree("F:\NatureRecognition/tree.txt")
print(tt)