import queue

import graphviz
import numpy as np

from Tree import nTree


class DesignTreeGenerate:
    def __init__(self, type):
        self.test_data = None
        self.node_code = 0
        self.type = type
        self.nTree = nTree()
        self.tempTree = nTree()
        self.tempTree_notCut = None

    # 计算信息熵
    def entropy(self, df):
        last_col = df.columns[-1]
        if len(df[last_col].unique()) > 1:
            count = int(df[last_col].value_counts()['yes'])
        else:
            return 0
        total = int(df.shape[0])
        pk1 = count / total
        pk2 = 1 - pk1
        ent = (pk1 * np.log2(pk1) + pk2 * np.log2(pk2)) * -1
        return ent

    # 计算IV（a）函数的值
    def IV(self, df):
        attribute = []
        Sum = 0
        total = len(df.columns)
        for i in range(total - 1):
            num = len(df[df.columns[i]].value_counts())
            for index in range(num):
                total = int(df.shape[0])
                temp = int(df[df.columns[i]].value_counts()[index])
                Proportion = temp / total
                Sum = Sum - Proportion * np.log2(Proportion)
            attribute.append(Sum)
        return attribute

    # 计算信息增益率
    def gain_ratio(self, df):
        attribute = self.entropy_gain(df)
        iv = self.IV(df)
        for i in range(len(attribute)):
            if iv[i] == 0:
                attribute[i] = 0
            else:
                attribute[i] = attribute[i] / iv[i]
        return attribute

    # 计算信息增益
    def entropy_gain(self, df):
        ent = self.entropy(df)
        total = len(df.columns)
        attribute = []
        for index in range(total - 1):
            num = len(df[df.columns[index]].value_counts())
            entSplit = 0
            for i in range(num):
                total = int(df.shape[0])
                temp = int(df[df.columns[index]].value_counts()[i])
                Proportion = temp / total
                row = df[df.columns[index]] == df[df.columns[index]].unique()[i]
                new_df = df[row]
                entSplit = entSplit + self.entropy(new_df) * Proportion
            attribute.append(ent - entSplit)
        return attribute

    # 寻找最大值
    def find_max(self, attribute):
        ans = []
        pos = 0
        iMax = attribute[0]
        for i in range(len(attribute)):
            if attribute[i] > iMax:
                iMax = attribute[i]
                pos = i
        ans.append(pos)
        ans.append(iMax)
        return ans

    # 计算基尼指数
    def gini(self, df):
        last_col = df.columns[-1]
        if len(df[last_col].unique()) > 1:
            count = int(df[last_col].value_counts()['yes'])
        else:
            return 0
        pk1 = count / len(df)
        pk2 = 1 - pk1
        gini = 1 - pk1 * pk1 - pk2 * pk2
        return gini

    # 计算Gini_index
    def gini_index(self, df):
        attribute = []
        for i in range(len(df.columns) - 1):
            num = len(df[df.columns[i]].value_counts())
            ans = 0
            for index in range(num):
                total = int(df.shape[0])
                temp = int(df[df.columns[i]].value_counts()[index])
                Proportion = temp / total
                row = df[df.columns[i]] == df[df.columns[i]].unique()[index]
                new_df = df[row]
                ans = ans + Proportion * self.gini(new_df)
            attribute.append(ans)
        return attribute

    # 寻找最小值
    def find_min(self, attribute):
        ans = []
        pos = 0
        iMin = attribute[0]
        for i in range(len(attribute)):
            if attribute[i] < iMin:
                iMin = attribute[i]
                pos = i
        ans.append(pos)
        ans.append(iMin)
        return ans

    # 构建决策树的详细步骤
    def Operate(self, node, max_gain, max_value, record, file, name):
        # 判断是否为叶子结点，如果是叶子结点，记录入.dot文件后直接返回，终止函数
        last_col = node.columns[-1]
        if len(node[last_col].unique()) == 1:
            file.write(str(self.node_code) + "[label = " + str(node[last_col].unique()[0]) + "]\n")
            print("当前结点样本都属于同一类别，无需划分，故将该节点" + str(node[last_col].unique()[0]) + "设置为叶节点")
            return 0
        # 不是叶子结点，开始获取结点需要填写的数据的内容
        feature = node.columns[max_gain]
        count = int(node[last_col].value_counts()['yes'])
        value = "yes: " + str(count) + "no: " + str(len(node) - count)
        # 记录非叶结点信息
        iGet_node = str(str(self.node_code) + '[label="feature = ' + str(feature) + r'\n' + name + ' = ' + str(
            max_value) + r'\n samples = ' + str(len(node)) + r'\n value = ' + value + '"]\n')
        iGet_trans = ""
        origin = self.node_code
        # 用于判断是否又被遗漏的分支
        for i in record:
            if i[0] == str(node.columns[max_gain]):
                flag = i[1]
        temp = flag
        # 记录根节点，根节点的node_code一定为0
        if origin == 0:
            self.nTree.insert(feature, node[feature].unique(), "", "")
            self.tempTree.insert(feature, node[feature].unique(), "", "")

        # 计算子节点
        if len(node[node.columns[max_gain]].unique()) == 1:
            print("样本子集中所有样本属性一样，无需划分")
            return
        for i in range(len(node[node.columns[max_gain]].unique())):
            # 如果会记录分支则将flag内容清空
            for index in range(len(flag)):
                if flag[index] == str(node[node.columns[max_gain]].unique()[i]):
                    flag[index] = ""
            # 根据属性值分类后，映射的数据集该属性值应统一
            row = node[node.columns[max_gain]] == node[node.columns[max_gain]].unique()[i]
            new_node = node[row]
            # 记录结点的编号
            self.node_code = self.node_code + 1
            # 记录子节点与父节点的关系，这些节点是父节点映射到的，记录他们的值
            iGet_trans = iGet_trans + str(origin) + "->" + str(
                self.node_code) + '[labeldistance=2.5, labelangle=-15, headlabel="' + str(
                node[node.columns[max_gain]].unique()[i]) + '"]\n'
            del new_node[feature]
            if len(new_node.columns) == 1:
                print("属性集为空停止划分")
                return
            temp_flag = 1
            for i in range(len(new_node.columns)):
                if len(new_node.iloc[i].unique()) != 1:
                    temp_flag = 0
                    break
            if temp_flag == 1:
                print('当前节点样本集合为空')
                return
            temp_flag = 1
            if (new_node.nunique(axis=1) == 1).all():
                print('当前属性集合取值相同')
                return

            # 根据记录的信息将值插入到三叉树中
            if len(new_node[last_col].unique()) == 1:
                feature01 = str(new_node[last_col].unique()[0])
                self.nTree.insert(str(new_node[last_col].unique()[0]), [],
                                  str(node[node.columns[max_gain]].unique()[i]), feature)
            # 不是叶子结点，开始获取结点需要填写的数据的内容
            else:
                attribute = self.gini_index(new_node)
                min_gain01 = self.find_min(attribute)[0]
                min_value01 = self.find_min(attribute)[1]
                feature01 = new_node.columns[min_gain01]
                count = int(new_node[last_col].value_counts()['yes'])
                self.tempTree = self.nTree
                self.tempTree.insert('no', temp,
                                     str(node[node.columns[max_gain]].unique()[i]), feature)
                self.nTree.insert(feature01, temp,
                                  str(node[node.columns[max_gain]].unique()[i]), feature)
                if len(new_node.columns) == 1:
                    print("当前属性集为空，无需划分")
                    return
            self.build_tree(new_node, file, record)
        # 信息写入.dot文件
        file.write(iGet_node + iGet_trans + "\n")
        print("满足规则，在" + feature + "结点后选择属性" + feature01 + "作为划分属性")
        # 补枝
        for i in flag:
            if i != "":
                self.node_code = self.node_code + 1
                file.write(str(self.node_code) + "[label = yes" + "]\n")
                self.nTree.insert('no', [],
                                  str(i), feature)
                file.write(str(origin) + "->" + str(
                    self.node_code) + '[labeldistance=2.5, labelangle=-15, headlabel="' + str(
                    i) + '"]\n')
                print("当前结点包含的样本集合为空，停止递归" + "它的父节点是" + feature)

    # 构建决策树的详细步骤(预剪枝)
    def OperateForCut(self, node, max_gain, max_value, record, file, name):
        # 判断是否为叶子结点，如果是叶子结点，记录入.dot文件后直接返回，终止函数
        last_col = node.columns[-1]
        if len(node[last_col].unique()) == 1:
            file.write(str(self.node_code) + "[label = " + str(node[last_col].unique()[0]) + "]\n")
            print("当前结点样本都属于同一类别，无需划分，故将该节点" + str(node[last_col].unique()[0]) + "设置为叶节点")
            return 0
        # 不是叶子结点，开始获取结点需要填写的数据的内容
        feature = node.columns[max_gain]
        count = int(node[last_col].value_counts()['yes'])
        value = "yes: " + str(count) + "no: " + str(len(node) - count)
        # 记录非叶结点信息
        iGet_node = str(str(self.node_code) + '[label="feature = ' + str(feature) + r'\n' + name + ' = ' + str(
            max_value) + r'\n samples = ' + str(len(node)) + r'\n value = ' + value + '"]\n')
        iGet_trans = ""
        origin = self.node_code
        # 用于判断是否又被遗漏的分支
        for i in record:
            if i[0] == str(node.columns[max_gain]):
                flag = i[1]
        temp = flag
        # 记录根节点，根节点的node_code一定为0
        if origin == 0:
            self.nTree.insert(feature, node[feature].unique(), "", "")
            self.tempTree.insert(feature, node[feature].unique(), "", "")

        # 计算子节点
        if len(node[node.columns[max_gain]].unique()) == 1:
            print("样本子集中所有样本属性一样，无需划分")
            return
        for i in range(len(node[node.columns[max_gain]].unique())):
            # 如果会记录分支则将flag内容清空
            for index in range(len(flag)):
                if flag[index] == str(node[node.columns[max_gain]].unique()[i]):
                    flag[index] = ""
            # 根据属性值分类后，映射的数据集该属性值应统一
            row = node[node.columns[max_gain]] == node[node.columns[max_gain]].unique()[i]
            new_node = node[row]
            # 记录结点的编号
            self.node_code = self.node_code + 1
            # 记录子节点与父节点的关系，这些节点是父节点映射到的，记录他们的值
            iGet_trans = iGet_trans + str(origin) + "->" + str(
                self.node_code) + '[labeldistance=2.5, labelangle=-15, headlabel="' + str(
                node[node.columns[max_gain]].unique()[i]) + '"]\n'

            # 根据记录的信息将值插入到三叉树中
            if len(new_node[last_col].unique()) == 1:
                feature01 = str(new_node[last_col].unique()[0])
                self.nTree.insert(str(new_node[last_col].unique()[0]), [],
                                  str(node[node.columns[max_gain]].unique()[i]), feature)
                self.build_tree(new_node, file, record)
            # 不是叶子结点，开始获取结点需要填写的数据的内容
            else:
                attribute = self.gini_index(new_node)
                min_gain01 = self.find_min(attribute)[0]
                min_value01 = self.find_min(attribute)[1]
                feature01 = new_node.columns[min_gain01]
                count = int(new_node[last_col].value_counts()['yes'])
                self.tempTree = self.nTree
                self.tempTree.insert('no', temp,
                                     str(node[node.columns[max_gain]].unique()[i]), feature)
                self.nTree.insert(feature01, temp,
                                  str(node[node.columns[max_gain]].unique()[i]), feature)
                del new_node[feature]
                if len(new_node.columns) == 1:
                    print("属性集为空停止划分")
                    return
                temp_flag = 1
                for i in range(len(new_node.columns)):
                    if len(new_node.iloc[i].unique()) != 1:
                        temp_flag = 0
                        break
                if temp_flag == 1:
                    print('当前节点样本集合为空')
                    return
                temp_flag = 1
                if (new_node.nunique(axis=1) == 1).all():
                    print('当前属性集合取值相同')
                    return

                # 判断是否需要预剪枝
                tag = self.pre_cut(feature)
                if not tag:
                    self.build_tree(new_node, file, record)
                else:
                    file.write(str(self.node_code) + "[label = " + str(node[last_col].unique()[0]) + "]\n")
        # 信息写入.dot文件
        file.write(iGet_node + iGet_trans + "\n")
        # 补枝
        for i in flag:
            if i != "":
                self.node_code = self.node_code + 1
                file.write(str(self.node_code) + "[label = yes" + "]\n")
                self.nTree.insert('no', [],
                                  str(i), feature)
                file.write(str(origin) + "->" + str(
                    self.node_code) + '[labeldistance=2.5, labelangle=-15, headlabel="' + str(
                    i) + '"]\n')
                print("当前结点包含的样本集合为空，停止递归" + "它的父节点是" + feature)

    # 建树，根据输入信息判断应该用哪一种算法实现
    def build_tree(self, node, file, record):
        if self.type == "entropy":
            attribute = self.entropy_gain(node)
            max_gain = self.find_max(attribute)[0]
            max_value = self.find_max(attribute)[1]
            self.Operate(node, max_gain, max_value, record, file, 'gain')
        if self.type == "c4.5":
            attribute = self.gain_ratio(node)
            max_gain = self.find_max(attribute)[0]
            max_value = self.find_max(attribute)[1]
            self.Operate(node, max_gain, max_value, record, file, 'gain_ratio')
        if self.type == "cart":
            attribute = self.gini_index(node)
            min_gain = self.find_min(attribute)[0]
            min_value = self.find_min(attribute)[1]
            self.Operate(node, min_gain, min_value, record, file, 'gini_index')
        if self.type == "预剪枝":
            attribute = self.gini_index(node)
            min_gain = self.find_min(attribute)[0]
            min_value = self.find_min(attribute)[1]
            self.OperateForCut(node, min_gain, min_value, record, file, 'gini_index')

    # 根据.dot文件绘制可视化决策树
    def write_dot(self, df):
        temp = []
        n = len(df.columns)
        for i in range(n):
            flag = []
            flag.append(str(df.columns[i]))
            flag.append(df[df.columns[i]].unique())
            temp.append(flag)
        if self.type == "entropy":
            with open('id3Tree.txt', 'w') as file:
                pass
            file = open("id3Tree.txt", "a")
            file.write("digraph Tree {\n")
            self.build_tree(df, file, temp)
            file.write("}")
        elif self.type == "c4.5":
            with open('c4.5Tree.txt', 'w') as file:
                pass
            file = open("c4.5Tree.txt", "a")
            file.write("digraph Tree {\n")
            self.build_tree(df, file, temp)
            file.write("}")
        elif self.type == "cart":
            with open('cartTree.txt', 'w') as file:
                pass
            file = open("cartTree.txt", "a")
            file.write("digraph Tree {\n")
            self.build_tree(df, file, temp)
            file.write("}")
        elif self.type == "预剪枝":
            with open('cutTree.txt', 'w') as file:
                pass
            file = open("cutTree.txt", "a")
            file.write("digraph Tree {\n")
            self.build_tree(df, file, temp)
            file.write("}")

    def plot(self):
        if self.type == 'entropy':
            with open("id3Tree.txt", encoding="GBK") as f:
                dot_graph = f.read()
            graph = graphviz.Source(dot_graph, encoding="GBK")
            graph.view()
        elif self.type == 'c4.5':
            with open("c4.5Tree.txt", encoding="GBK") as f:
                dot_graph = f.read()
            graph = graphviz.Source(dot_graph, encoding="GBK")
            graph.view()
        elif self.type == 'cart':
            with open("cartTree.txt", encoding="GBK") as f:
                dot_graph = f.read()
            graph = graphviz.Source(dot_graph, encoding="GBK")
            graph.view()
        elif self.type == '预剪枝':
            with open("cutTree.txt", encoding="GBK") as f:
                dot_graph = f.read()
            graph = graphviz.Source(dot_graph, encoding="GBK")
            graph.view()

    def pre_cut(self, label):
        last_col = self.test_data.columns[-1]
        judge = 0
        for i in range(len(self.test_data)):
            if self.nTree is not None:
                if self.test_data[last_col][i] == self.nTree.predict(self.nTree, self.test_data.loc[i]):
                    judge = judge + 1
            else:
                if self.test_data[last_col][i] == 'no':
                    judge = judge + 1
        percent01 = judge / len(self.test_data)
        judge01 = 0
        for i in range(len(self.test_data)):
            if self.tempTree is not None:
                if self.test_data[last_col][i] == self.tempTree.predict(self.tempTree, self.test_data.loc[i]):
                    judge01 = judge01 + 1
            else:
                if self.test_data[last_col][i] == 'no':
                    judge01 = judge01 + 1
        percent02 = judge01 / len(self.test_data)
        if percent01 < percent02:
            print("对于分支结点" + label + "剪枝前准确率：" + str(percent02) + ", 剪枝后准确率：" + str(percent01) + "-> 不剪枝")
            return False
        else:
            print("对于分支结点" + label + "剪枝前准确率：" + str(percent02) + ", 剪枝后准确率：" + str(percent01) + "-> 剪枝")
            return True

    def set_test_data(self, test_data):
        self.test_data = test_data
