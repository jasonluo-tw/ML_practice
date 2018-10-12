import numpy as np
from prepare_data import iris_datasets, standardize
from plot_dec_regions import plot_decision_regions
from ML_algorithm import LR_method, SVM_method, decision_tree, random_forest, knn
import matplotlib.pyplot as plt

## prepare data
X_train, X_test, y_train, y_test = iris_datasets()

## standardize
X_train_std, X_test_std = standardize(X_train, X_test)

# combine train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

methods = []
names = ['Logistic Regression', 'SVM(linear)', 'SVM(rbf)', 'Decision Tree', 'Random forest', 'Knn']

methods.append(LR_method())
methods.append(SVM_method())
methods.append(SVM_method('rbf'))
methods.append(decision_tree())
methods.append(random_forest(nums=20))
methods.append(knn())

## start the loop
plt.rcParams['axes.unicode_minus']=False
plt.figure()
for idx, model in enumerate(methods):
    model.fit(X_train_std, y_train)
    
    ## plot the result
    plt.subplot(2, round(len(methods)/2), idx+1)
    plot_decision_regions(X_combined_std, y_combined, classifier=model, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.title(names[idx])

plt.show()
