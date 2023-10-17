from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np
from DecisionTree import DesignTreeGenerate

'''
Task 1
1.1 & 1.2 & 1.3导入数据集并完成id3及C4.5决策树的构建与可视化(C4.5用ID3近似)
'''
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建ID3决策树模型
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# 使用sklearn的export_graphviz导出决策树
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True)

# 使用graphviz库生成可视化决策树
graph = graphviz.Source(dot_data)
graph.render("iris_id3")

'''
Task 1
1.4 CART决策树的构建与可视化
'''
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建ID3决策树模型
clf = DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X, y)

# 使用sklearn的export_graphviz导出决策树
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True)

# 使用graphviz库生成可视化决策树
graph = graphviz.Source(dot_data)
graph.render("iris_CART")

'''
Task 2 
自行实现ID3, C4.5, CART算法
'''

data = [
    ['green', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['black', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['black', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['green', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['white', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['green', 'slightly coiled', 'dull', 'clear', 'slightly concave', 'soft sticky', 'yes'],
    ['black', 'slightly coiled', 'dull', 'slightly blurry', 'slightly concave', 'soft sticky', 'yes'],
    ['black', 'slightly coiled', 'dull', 'clear', 'slightly concave', 'hard', 'yes'],
    ['black', 'slightly coiled', 'dull', 'slightly blurry', 'slightly concave', 'hard', 'no'],
    ['green', 'hard', 'crisp', 'clear', 'flat', 'soft sticky', 'no'],
    ['white', 'hard', 'crisp', 'blurry', 'flat', 'hard', 'no'],
    ['white', 'coiled', 'dull', 'blurry', 'flat', 'soft sticky', 'no'],
    ['green', 'slightly coiled', 'dull', 'slightly blurry', 'concave', 'hard', 'no'],
    ['white', 'slightly coiled', 'dull', 'slightly blurry', 'concave', 'hard', 'no'],
    ['black', 'slightly coiled', 'dull', 'clear', 'slightly concave', 'soft sticky', 'no'],
    ['white', 'coiled', 'dull', 'blurry', 'flat', 'hard', 'no'],
    ['green', 'coiled', 'dull', 'slightly blurry', 'slightly concave', 'hard', 'no']
]

df = pd.DataFrame(data, columns=['color', 'root', 'knock', 'texture', 'navel', 'touch', 'good_melon'])

while True:
    iType = input("请输入你想要使用的算法：")
    entropyTree = DesignTreeGenerate(iType)
    entropyTree.write_dot(df)
    entropyTree.plot()
    if iType == "end":
        break

'''
Task 3 
实现预剪枝
'''

train_data = [
    ['green', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['black', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['black', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['green', 'slightly coiled', 'dull', 'clear', 'slightly concave', 'soft sticky', 'yes'],
    ['black', 'slightly coiled', 'dull', 'slightly blurry', 'slightly concave', 'soft sticky', 'yes'],
    ['green', 'hard', 'crisp', 'clear', 'flat', 'soft sticky', 'no'],
    ['white', 'slightly coiled', 'dull', 'slightly blurry', 'concave', 'hard', 'no'],
    ['black', 'slightly coiled', 'dull', 'clear', 'slightly concave', 'soft sticky', 'no'],
    ['white', 'coiled', 'dull', 'blurry', 'flat', 'hard', 'no'],
    ['green', 'coiled', 'dull', 'slightly blurry', 'slightly concave', 'hard', 'no']
]
train_df = pd.DataFrame(train_data, columns=['color', 'root', 'knock', 'texture', 'navel', 'touch', 'good_melon'])

test_data = [
    ['green', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['white', 'coiled', 'dull', 'clear', 'concave', 'hard', 'yes'],
    ['black', 'slightly coiled', 'dull', 'clear', 'slightly concave', 'hard', 'yes'],
    ['black', 'slightly coiled', 'dull', 'slightly blurry', 'slightly concave', 'hard', 'no'],
    ['white', 'hard', 'crisp', 'blurry', 'flat', 'hard', 'no'],
    ['white', 'coiled', 'dull', 'blurry', 'flat', 'soft sticky', 'no'],
    ['green', 'slightly coiled', 'dull', 'slightly blurry', 'concave', 'hard', 'no'],
]
test_df = pd.DataFrame(test_data, columns=['color', 'root', 'knock', 'texture', 'navel', 'touch', 'good_melon'])

entropyTree = DesignTreeGenerate("预剪枝")
entropyTree.set_test_data(test_df)
entropyTree.write_dot(train_df)
entropyTree.plot()
