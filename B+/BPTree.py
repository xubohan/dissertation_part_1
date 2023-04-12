from __future__ import annotations

class Node:
    def __init__(self, degree):
        self.degree:int = degree
        self.parent: Node = None
        self.keys = list()
        self.values = list()

    def split(self) -> Node:
        mid:int = self.degree // 2
        left = Node(self.degree)
        right = Node(self.degree)
        
        left.parent = self
        right.parent = self
        
        left.keys = self.keys[:mid]
        left.values = self.values[:mid+1]
        right.keys = self.keys[mid+1:]
        right.values = self.values[mid+1:]
        
        self.values = [left, right]
        self.keys = list(self.keys[mid])
        
        for child in left.values:
            if isinstance(child, Node):
                child.parent = left
        for child in right.values:
            if isinstance(child, Node):
                child.parent = right
        return self
    
    @property
    def size(self)-> int:
        return len(self.keys)
    
    @property
    def getKeySize(self)-> int:
        return len(self.keys)
    
    @property
    def getValueSize(self)-> int:
        return len(self.values)
    
    @property
    def isKeyEmpty(self)-> bool:
        return len(self.keys) == 0
    
    @property
    def isValueEmpty(self)-> bool:
        return len(self.values) == 0
    
    @property
    def isKeyFull(self) -> bool:
        return len(self.keys) == self.degree - 1
    
    @property
    def isValueFull(self) -> bool:
        return len(self.values) == self.degree 
    
    @property
    def isOneKeyLeft(self) -> bool:
        return len(self.keys) <= int(self.degree // 2)

    @property
    def reachLimit(self) -> bool:
        return len(self.keys) <= int(self.degree // 2) - 1

    @property
    def isRoot(self) -> bool:
        return self.parent is None


class LeafNode(Node):
    def __init__(self, degree):
        super().__init__(degree)
        self.prev: LeafNode = None
        self.next: LeafNode = None
    
    def insert(self, key, value):
        # return {int} for calculating
        # the number of keys
        if not self.keys:
            # if node is not exist
            self.keys.append(key)
            self.values.append(list(value))
            return 1
        for idx,key_val in enumerate(self.keys):
            # if key is the same, the value will append the value
            if key == key_val:
                # use List to store the timestamp here
                self.values[idx].append(value)
                return 0
            
            # add it in front of key_val
            elif key < key_val:
                self.keys.insert(idx,key)
                # self.keys = self.keys[:idx] + [key] + self.keys[idx:]
                self.values = self.values[:idx] + [[value]] + self.values[idx:]
                return 1
            # if key is not found, add it to the last position
            elif idx + 1 == self.size:
                self.keys.append(key)
                self.values.append([value])
                return 1
    
    def split(self) ->Node:
        # self == left LeafNode
        # LeafNode looks like
        # creating linked list
        mid:int = self.degree // 2
        top = Node(self.degree)
        right = LeafNode(self.degree)
        self.parent = top
        right.parent = top
        right.keys = self.keys[mid:]
        right.values = self.values[mid:]
        right.prev = self
        right.next = self.next
        
        top.values = [self, right]
        top.keys = list(right.keys[0])
        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        self.next = right
        return top
        
class BPTree(object):
    # default value could be larger
    def __init__(self, degree=9):
        # root should be LeafNode to store data
        self.root: Node = LeafNode(degree)
        self.degree: int = degree
        self.__count_keys: int = 0
    
    def __str__(self):
        node = self.head_leafnode
        if not node:
            return None

        temp = ''
        while node:
            for i in range(node.getValueSize):
                temp += str(node.values[i]) + ' -> '
            node = node.next
        return 'Value List: '+temp +'END'
        
    
    def insert(self, key, value):
        node = self.root
        while not isinstance(node, LeafNode):
            node, idx = self.inner_check(node, key)

        self.__count_keys += node.insert(key, value)
        while node.size == node.degree:  
            if not node.isRoot:
                parent = node.parent
                node = node.split()  
                _, idx = self.inner_check(parent, node.keys[0])
                
                parent.values.pop(idx)
                headnode = node.keys[0]

                for node_val in node.values:
                    if isinstance(node_val, Node):
                        node_val.parent = parent

                for idx, par_val in enumerate(parent.keys):
                    if headnode < par_val:
                        parent.keys.insert(idx, headnode)
                        # parent.keys = parent.keys[:idx] + [headnode] + parent.keys[idx:]
                        parent.values = parent.values[:idx] + node.values + parent.values[idx:]
                        break

                    elif idx + 1 == parent.size:
                        parent.keys.append(headnode)
                        parent.values.extend(node.values)
                        break   
                node = parent
            else:
                node = node.split()  
                self.root = node

    def inner_check(self,node: Node, key):
        for idx,key_val in enumerate(node.keys):
            if key < key_val:
                return (node.values[idx],idx)
            elif idx+1 == node.size:
                return (node.values[idx+1], idx+1)

    def search(self, key):
        node = self.root

        while not isinstance(node, LeafNode):
            node, _ = self.inner_check(node, key)

        for idx, key_val in enumerate(node.keys):
            if key == key_val:
                return node.values[idx]
        return None
    
    @property
    def count_keys(self):
        return self.__count_keys
        
    @property
    def head_leafnode(self):
        if not self.root:
            return None

        node = self.root
        while not isinstance(node, LeafNode):
            node = node.values[0]

        return node
            
            

if __name__ == '__main__':
    from random import randint
    import datetime
    import numpy as np
    import pandas as pd
    import os
    # print(os.path.abspath('..'))
    data = pd.read_csv(os.path.join(os.path.abspath('..')+'/dataset', 'sx-stackoverflow.txt'), header=None, sep = ' ')
    data = data.rename(columns = {0:'source', 1:'target', 2:'timestamp'})

    bpt = BPTree(degree=5)
    testi = bpt.insert
    c = datetime.datetime.now()
    for i in range(10000000):
        testi((data['source'][i],data['target'][i]),data['timestamp'][i])
    print('BPT time spend: ',datetime.datetime.now() -c, 'key counts: ', bpt.count_keys)


    bpt = BPTree(degree=5)
    testi = bpt.insert
    c = datetime.datetime.now()
    a = map(lambda x,y,z : testi((x,y), z),data['source'][:10000000],data['target'][:10000000],data['timestamp'][:10000000])
    list(a)
    print('BPT time spend: ',datetime.datetime.now() -c, 'key counts: ', bpt.count_keys)


    # c = datetime.datetime.now()
    # a = BPTree(degree=5)
    # t = a.insert
    # for b in range(10000000):
    #     t((randint(0,100),randint(0,100)),randint(0, 100000))
    # print('timespend: ',datetime.datetime.now()-c)
    # # print(a)
    # print('Value:',a.search((12,944444)))


