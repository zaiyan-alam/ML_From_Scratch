import numpy as np

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size   #number of unique classes
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True
        self.dim_split = None  # the index of the feature to be split
        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):

        def weighted_entropy(branches):
            arr = np.array(branches)
            net = np.sum(arr, axis=0)
            frac = net / np.sum(net)
            ent = arr/net
            ent = np.array([[-i * np.log2(i) if i > 0 else 0 for i in z] for z in ent])
            ent = np.sum(ent, axis=0)
            ent = np.sum(ent * frac)
            return ent

        for index_d in range(len(self.features[0])):
            if not 'minimum_ent' in locals():
                minimum_ent = float('inf')
            temp = np.array(self.features)[:, index_d]
            if None in temp:
                continue
            brch_val = np.unique(temp)
            brch = np.zeros((self.num_cls, len(brch_val)))
            for i, val in enumerate(brch_val):
                y = np.array(self.labels)[np.where(temp == val)]
                for yi in y:
                    brch[yi, i] += 1
            ent = weighted_entropy(brch)
            if ent < minimum_ent:
                minimum_ent = ent
                self.dim_split = index_d
                self.feature_uniq_split = brch_val.tolist()

        temp1 = np.array(self.features)[:, self.dim_split]
        x = np.array(self.features, dtype=object)
        x[:, self.dim_split] = None
        for each in self.feature_uniq_split:
            index_list = np.where(temp1 == each)
            new_val_x = x[index_list].tolist()
            new_val_y = np.array(self.labels)[index_list].tolist()
            child = TreeNode(new_val_x, new_val_y, self.num_cls)
            if np.array(new_val_x).size == 0 or all(v is None for v in new_val_x[0]):
                child.splittable = False
            self.children.append(child)
        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()
        return

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            # print(feature)
            child_index = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[child_index].predict(feature)
        else:
            return self.cls_max
