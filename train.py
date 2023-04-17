import ast
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sns.set_theme(style="dark")


class GetFeatures(ast.NodeVisitor):
    """
    Methods of this class allow to get additional features from the code to create a feature object matrix.

    Features : number of functions, total length of all function documentations (docs),
               total length of all class docs, total length of all class names,
               total length of all function names, max depth of conditions & cycles in the code
               and average density of the string
    """


    def __init__(self, code):
        """
        Contains features described above
        """

        self.Functions = 0
        self.FunctionDocsLen = 0
        self.ClassDocsLen = 0
        self.ClassNamesLen = 0
        self.FunctionNamesLen = 0
        self.MaxDepth = 0
        self.AvgDensity = np.mean([len(row) for row in code.split("\n")]) / len(code.split("\n"))


    def _get_all_depths(self, node):
        """
        Finds the max depth feature

        Parameters:
            node (ast.FunctionDef or ast.If or ast.For or ast.While) : the certain node in the syntax tree

        Returns:
            depths (list) : a list of all possible depths in the given node
        """

        children = [n for n in ast.iter_child_nodes(node) if isinstance(n, (ast.If, ast.For, ast.While))]
        if children:
            depths = [1] * len(children)
            for i, subnode in enumerate(children):
                depths[i] += max(self._get_all_depths(subnode))
            return depths

        return [0]


    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Defines rules to do in the nodes of the ast.ClassDef type

        Returns:
            None
        """

        self.ClassNamesLen += len(node.name)

        if ast.get_docstring(node) is not None:
            self.ClassDocsLen += len(ast.get_docstring(node).strip())

        for n in ast.iter_child_nodes(node):
            if isinstance(n, ast.FunctionDef):
                self.visit_FunctionDef(n)


    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Defines rules to do in the nodes of the ast.FunctionDef type

        Returns:
            None
        """

        self.Functions += 1
        self.FunctionNamesLen += len(node.name)

        if ast.get_docstring(node) is not None:
            self.FunctionDocsLen += len(ast.get_docstring(node).strip())

        depth = max(self._get_all_depths(node))
        if depth > self.MaxDepth:
            self.MaxDepth = depth


    def _get_features(self):
        """
        Collects all features of the code after parsing

        Returns:
            features (nd.array) : a numpy array of all features
        """

        return np.array([self.Functions,
                         self.ClassDocsLen,
                         self.ClassNamesLen,
                         self.FunctionDocsLen,
                         self.FunctionNamesLen,
                         self.MaxDepth,
                         self.AvgDensity])


def get_directories():
    """
    Allows user to run this code from the console terminal
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str)
    parser.add_argument("plagiat1", type=str)
    parser.add_argument("plagiat2", type=str)
    parser.add_argument("--model", type=str)
    
    return parser.parse_args()


def main():
    np.random.seed(42)

    dirs = get_directories()

    X, y = [], []
    for dir in vars(dirs):
        if dir == "model":
            break
        y += [int(dir == dirs.files)] * len(os.listdir(dir))
        for file in os.listdir(dir):
            code = open(dir + "\\" + file, "r", encoding="utf-8").read()
            tree = ast.parse(code)
            analyzer = GetFeatures(code)
            analyzer.visit(tree)
            X.append(analyzer._get_features())

    X, y = np.array(X), np.array(y)
    X = X.reshape(y.shape[0], 7)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True)

#    scaler = StandardScaler()

#    X_train = scaler.fit_transform(X_train, y_train)
#    X_test = scaler.transform(X_test)

    models = [RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42), XGBClassifier(n_estimators=14, max_depth=5, random_state=42), GradientBoostingClassifier(n_estimators=14, max_depth=5, random_state=42), KNeighborsClassifier(16, p=2)]
    model_names = ["Random Forest", "XGBoost", "GradientBoosting", "KNeighbours"]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharey=True, layout="constrained")
    axs[0][0].set_ylabel("False Positive Rate, FPR")
    axs[1][0].set_ylabel("False Positive Rate, FPR")
    for i, model in enumerate(models):
        start = time()
        model.fit(X_train, y_train)
        stop = time()
        y_pred = model.predict_proba(X_test)[:,1]

        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred)
        fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train)[:,1])

        axs[i // 2, i % 2].plot(fpr_test, tpr_test, linewidth=2, label=f"test data\nROC_AUC={round(roc_auc_score(y_test, y_pred), 4)}")
        axs[i // 2, i % 2].plot(fpr_train, tpr_train, linewidth=2, label=f"train data\nROC_AUC={round(roc_auc_score(y_train, model.predict_proba(X_train)[:,1]), 4)}")
        axs[i // 2, i % 2].text(x=0.9, y=0.3, s=f"Fitting time = {round((stop - start) * 1000, 2)} ms", ha="center", va="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
        axs[i // 2, i % 2].set_xlabel("True Positive Rate, TPR")
        axs[i // 2, i % 2].plot([0, 1], [0, 1], 'k--')
        axs[i // 2, i % 2].grid(True)
        axs[i // 2, i % 2].legend(loc="lower right")
        axs[i // 2, i % 2].set_title(f'ROC curve for {model_names[i]}')
    plt.savefig("ROC.png")
    pickle.dump(models[1], open(dirs.model, "wb"))


if __name__ == "__main__":
    main()
