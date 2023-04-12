from __future__ import annotations
'''
@author: Bohan Xu
'''
class Node:
    def __init__(self, order):
        self.order:int = order
        self.parent: Node = None
        self.keys = list()
        self.values = list()

    def split(self) -> Node:
        '''
        Calculate the middle index $mid$ of the node. if the order is notdivisible, the mid will round down
        Create two non-leaf nodes, $left$ and $right$, with the same order as the original node
        Set the parent of both new nodes to be the original node
        Distribute the keys $x.keys$ and the pointer $x.values$ from the original node to the $left$ and $right$ non-leaf nodes. 
        The $left$ node receives the keys $x.keys$ and the pointer $x.values$ before the middle index $mid$; 
        and the $right$ node receives the keys $x.keys$ and the pointer $x.values$ after the middle index $mid$
        Update the pointer of the original node to contain the two non-leaf nodes, $left$ and $right$
        Update the key of the original node to only contain the key $x.keys$ at the middle index $mid$
        Use $for$ loop to update the parent pointers for all children in the $left$ and $right$ non-leaf nodes
        '''
        mid:int = self.order // 2
        left = Node(self.order)
        right = Node(self.order)
        
        left.parent = self
        right.parent = self
        
        left.keys = self.keys[:mid]
        left.values = self.values[:mid+1]
        right.keys = self.keys[mid+1:]
        right.values = self.values[mid+1:]
        
        self.values = [left, right]
        self.keys = [self.keys[mid]]
        
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
        return len(self.keys) == self.order - 1
    
    @property
    def isValueFull(self) -> bool:
        return len(self.values) == self.order 
    
    @property
    def isOneKeyLeft(self) -> bool:
        return len(self.keys) <= int(self.order // 2)

    @property
    def reachLimit(self) -> bool:
        return len(self.keys) <= int(self.order // 2) - 1

    @property
    def isRoot(self) -> bool:
        return self.parent is None


class LeafNode(Node):
    def __init__(self, order):
        super().__init__(order)
        self.prev: LeafNode = None
        self.next: LeafNode = None
    
    def insert(self, key, value):
        # return {int} for calculating
        # the number of keys
        if not self.keys:
            # if node is not exist
            self.keys.append(key)
            self.values.append([value])
            return 1
        for idx,key_val in enumerate(self.keys):
            # if key is the same, the value will append the value
            if key == key_val:
                # use List to store the timestamp here
                self.values[idx].append(value)
                return 0
            
            # add it in front of key_val
            elif key < key_val:
                self.keys = self.keys[:idx] + [key] + self.keys[idx:]
                self.values = self.values[:idx] + [[value]] + self.values[idx:]
                return 1
            # if key is not found, add it to the last position
            elif idx + 1 == self.size:
                self.keys.append(key)
                self.values.append([value])
                return 1
    
    def split(self) ->Node:
        '''
        Calculate the $mid$ value, which is the order divided by 2. If the order is not divisible, the $mid$ will round down
        Create a non-leaf node $top$ and a leaf node $right$
        Set the current leaf node as left child of the $top$ node, and set the $right$ leaf node as the right child of the $top$ node 
        Assign the keys and values to the right of the middle position ($mid$) of the current leaf node to the $right$ leaf node
        Update the $prev$ leaf node of $right$ to be the current leaf node, and update the $next$ leaf node of $right$ to be the next leaf node of the current leaf node
        Set the key of the $top$ node to be the first key of the $right$ leaf node
        Update the keys and values of the current leaf node, and the current leaf node only keeps the $x.keys[:mid]$ and $x.values[:mid]$
        Update the $next$ leaf node of the current leaf node to the $right$ leaf node

        '''
        # self == left LeafNode
        # LeafNode looks like
        # creating linked list
        mid:int = self.order // 2
        top = Node(self.order)
        right = LeafNode(self.order)
        self.parent = top
        right.parent = top
        right.keys = self.keys[mid:]
        right.values = self.values[mid:]
        right.prev = self
        right.next = self.next
        
        top.keys = [right.keys[0]]
        top.values = [self, right]

        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        self.next = right
        return top
        
class BPTree(object):
    # default value could be larger
    def __init__(self, order=9):
        # root should be LeafNode to store data
        self.root: Node = LeafNode(order)
        self.order: int = order
        self.__count_keys: int = 0
        self.op_times: int = 0
    
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
            node, idx = self.__inner_check(node, key)

        self.__count_keys += node.insert(key, value)
        self.op_times += 1
        while node.size == node.order:  
            if not node.isRoot:
                parent = node.parent
                node = node.split()  
                _, idx = self.__inner_check(parent, node.keys[0])
                
                parent.values.pop(idx)
                headnode = node.keys[0]

                for node_val in node.values:
                    if isinstance(node_val, Node):
                        node_val.parent = parent

                for idx, par_val in enumerate(parent.keys):
                    if headnode < par_val:
                        parent.keys.insert(idx, headnode)
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

    def __inner_check(self,node: Node, key):
        for idx,key_val in enumerate(node.keys):
            if key < key_val:
                return (node.values[idx],idx)
            elif idx+1 == node.size:
                return (node.values[idx+1], idx+1)

    def search(self, key):
        node = self.root

        while not isinstance(node, LeafNode):
            node, _ = self.__inner_check(node, key)

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

    bpt = BPTree(order=5)
    testi = bpt.insert
    c = datetime.datetime.now()
    for i in range(10000000):
        testi((data['source'][i],data['target'][i]),data['timestamp'][i])
    print('BPT time spend: ',datetime.datetime.now() -c, 'key counts: ', bpt.count_keys)


    bpt = BPTree(order=5)
    testi = bpt.insert
    c = datetime.datetime.now()
    a = map(lambda x,y,z : testi((x,y), z),data['source'][:10000000],data['target'][:10000000],data['timestamp'][:10000000])
    list(a)
    print('BPT time spend: ',datetime.datetime.now() -c, 'key counts: ', bpt.count_keys)


    # c = datetime.datetime.now()
    # a = BPTree(order=5)
    # t = a.insert
    # for b in range(10000000):
    #     t((randint(0,100),randint(0,100)),randint(0, 100000))
    # print('timespend: ',datetime.datetime.now()-c)
    # # print(a)
    # print('Value:',a.search((12,944444)))


