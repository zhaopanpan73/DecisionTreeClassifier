# -*- coding: utf-8 -*-
from math import log
# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for  featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']  # 这里是属性的名字
    return  dataSet,labels

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if  featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
     numFeatrues=len(dataSet[0])-1  # 标签的个数
     baseEntropy=calcShannonEnt(dataSet)
     bestInfoGain=0.0
     bestFeature=-1
     for i in range(numFeatrues):
         featList=[example[i] for example in dataSet]  # 第i列有多少个取值,也就是对于某一个特定的特征，这个特征的不同取值有哪些,可能包含重复
         uniqueVals=set(featList)# 对上面的取值进行去重
         newEntropy=0.0
         for value in uniqueVals:
             # 取出某一个特征值的数据集
             subDataSet=splitDataSet(dataSet,i,value)
             prob=len(subDataSet)/float(len(dataSet))# 对于取某一特征取值，计算它对于总数据集的熵
             newEntropy+=prob*calcShannonEnt(subDataSet) # 计算子数据集的熵*子数据集所占的比例
             # 思考这里的subDataSet的实现方式是，得到当前特的当前取值所在的行。然后去掉取值为给定值的一列，即挖空了特征取值为给定值的那一列
         infoGain=baseEntropy-newEntropy
         if (infoGain>bestInfoGain):
             bestInfoGain=infoGain
             bestFeature=i
     return bestFeature


# 多数表决的方法
import operator
def majorityCnt(classList):  # 此时的classList是没有顺序的
    classCount={}
    for vote in classList:
        if vote not in classCount:
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]   # 返回的是类别而不是数目
# 代码示例
# >>> import operator
# >>> dd={"a":10,"b":90}
# >>> sortedClassCount=sorted(dd.iteritems(),key=operator.itemgetter(1),reverse=True)
# >>> ret=sortedClassCount=sorted(dd.iteritems(),key=operator.itemgetter(1),reverse=True)
# >>> ret
# [('b', 90), ('a', 10)]
# >>> ret[0][0]
# 'b'
# >>>
# >>> ret=sortedClassCount=sorted(dd.iteritems(),key=operator.itemgetter(0),reverse=True)
# >>> ret
# [('b', 90), ('a', 10)]
# >>> getcount=operator.itemgetter(0)  取第一个参数操作
# >>> map(getcount,dd)
# ['a', 'b']


# 递归的构建决策树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):# 说明标签里面仅有一个类别了
        return classList[0]
    if len(dataSet[0])==1 :   # 说明此时特征已经遍历完了，但是为什么是1呢
        return  majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    # 对目前最好的分隔特征进行记录和再分割
    # 用来存储树结构
    myTree={bestFeatLabel:{}}
    # 从label中删除当前的最优特征值，供后续调用
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:] # 为了保证在每次调用creatTree时不改变原列表的内容
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# 构造注解树，这才是我今天新接触的知识

# 获取叶节点的数目和树的层数  这个代码思想值得借鉴
# 获取叶节点的数目
# {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# 树的深度优先搜索
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=myTree.keys()[0]  # 当前树的根节点
    secondDict=myTree[firstStr]  # 树的几个孩子
    for key in secondDict.keys():  # {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
        # key 就是两个分支的判断取值
        # seconDict[key]对应的是 叶节点 或者判断节点
        if type(secondDict[key]==dict):
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs=1
    return numLeafs

# 树的深度优先搜索
def getTreeDepth(myTree):  # 第一个字典
    maxDepth=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr] # 第二个字典
    for key in secondDict.keys():
        if  type(secondDict[key]) ==dict:
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        # 比较每一个字节点高度的大小
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return  maxDepth

# 测试算法：是用决策树执行分类
# 只要涉及到树的都是用递归做的，，，，
def classify(inputTree,featLabels,testVec):
    firstStr=inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key])==dict:
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return  classLabel


# 树的存储
def storeTree(inputTree,filename):
    import pickle
    with open(filename,'w') as f:
        pickle.dump(inputTree,f)

# 树的读取
def grabTree(filename):
    import pickle
    with open(filename, 'r') as f:
       return  pickle.load(f)