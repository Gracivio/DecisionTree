class Node:
    def __init__(self, label, attribute):
        self.label = label
        self.child = []
        self.attribute = attribute
        for i in range(len(attribute)):
            self.child.append(None)

    def isLeaf(self):
        if self.label == 'yes' or self.label == 'no':
            return True


class nTree:
    def __init__(self):
        self.root = None
        self.current = None

    def insert(self, label, attribute, headLabel, father):
        if self.root is None:
            self.root = Node(label, attribute)
            return
        else:
            self._insert(self.root, label, attribute, headLabel, father)

    def _insert(self, current, label, attribute, headLabel, father):
        if current.label == 'yes' or current.label == 'no':
            return
        if current.label == father:
            for index in range(len(current.attribute)):
                if current.attribute[index] == headLabel:
                    current.child[index] = Node(label, attribute)
        for index in range(len(current.attribute)):
            if current.child[index]:
                self._insert(current.child[index], label, attribute, headLabel, father)

    def predict(self, tree, df):
        if tree.root is None:
            return None
        current = tree.root
        while not current.isLeaf():
            for i in range(len(current.attribute)):
                attribute = len(current.attribute) - 1
                if df[current.label] == current.attribute[i]:
                    attribute = i
                    break
            for i in range(len(current.attribute)):
                if attribute == i:
                    current = current.child[i]
            if current is None:
                return 'no'
        return current.label







