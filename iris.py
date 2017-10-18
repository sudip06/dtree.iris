from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def visualise(tree_clf, iris):
    from sklearn.tree import export_graphviz
    import pydot
    export_graphviz(
            tree_clf,
            out_file="iris_tree.dot",
            feature_names=iris.feature_names[:],
            class_names=iris.target_names,
            rounded=True, #rounds the text box
            filled=True #adds color
            )
    #In order for next line to work, we have to install graphviz via apt-get, else will get exception "Program dot not found in path." 
    (graph,) = pydot.graph_from_dot_file("iris_tree.dot")
    graph.write_png('iris_tree.png')


iris=load_iris()

'''
iris is type:sklearn.utils.Bunch
In [3]: dir(iris)
Out[3]: ['DESCR', 'data', 'feature_names', 'target', 'target_names']
'''
X=iris.data[:,:]
y=iris.target
tree_clf=DecisionTreeClassifier(max_depth=4)
tree_clf.fit(X,y)

visualise(tree_clf, iris)

