import numpy as np

def LR_method(re_pa=1000):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=re_pa, random_state=0)
    return lr

def SVM_method(ker='linear', re_pa=1.0):
    from sklearn.svm import SVC
    svm = SVC(kernel=ker, C=re_pa, random_state=0)
    return svm

def sgd_classifier():
    from sklearn.linear_model import SGDClassifier
    svm = SGDClassifier(loss='hinge')  # can choose perceptron or log
    return svm

def decision_tree(cri='entropy', tree_depth=3):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion=cri, max_depth=tree_depth, random_state=0)
    return tree

def random_forest(cri='entropy', nums=10):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(criterion=cri, n_estimators=nums, random_state=1, n_jobs=2)
    return forest

def knn(votes=5, dis=2):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=votes, p=dis, metric='minkowski')
    return knn
