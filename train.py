import ast
import argparse
import os
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

np.random.seed(42)

class GetFeatures(ast.NodeVisitor):


    def __init__(self, code):
        self.Functions = 0
        self.FunctionDocsLen = 0
        self.ClassDocsLen = 0
        self.ClassNamesLen = 0
        self.FunctionNamesLen = 0
        self.MaxDepth = 0
        self.MeanDensity = np.mean([len(row) for row in code.split("\n")]) / len(code.split("\n"))


    def _get_all_depths(self, node):
        children = [n for n in ast.iter_child_nodes(node) if isinstance(n, (ast.If, ast.For, ast.While))]
        if children:
            depths = [1] * len(children)
            for i, subnode in enumerate(children):
                depths[i] += max(self._get_all_depths(subnode))
            return depths

        return [0]


    def visit_ClassDef(self, node: ast.ClassDef):
        self.ClassNamesLen += len(node.name)

        if ast.get_docstring(node) is not None:
            self.ClassDocsLen += len(ast.get_docstring(node).strip())

        for n in ast.iter_child_nodes(node):
            if isinstance(n, ast.FunctionDef):
                self.visit_FunctionDef(n)


    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.Functions += 1
        self.FunctionNamesLen += len(node.name)

        if ast.get_docstring(node) is not None:
            self.FunctionDocsLen += len(ast.get_docstring(node).strip())

        depth = max(self._get_all_depths(node))
        if depth > self.MaxDepth:
            self.MaxDepth = depth


parser = argparse.ArgumentParser()
parser.add_argument("files", type=str)
parser.add_argument("plagiat1", type=str)
parser.add_argument("plagiat2", type=str)
parser.add_argument("--model")
dirs = parser.parse_args()

X, y = [], []
for dir in vars(dirs):
    if dir == "model":
        break
    y += [int(dir == dirs.files)] * len(os.listdir(dir))
    for file in os.listdir(dir):
        code = open(dir + "\\" + file, "r", encoding="utf-8").read()
        tree = ast.parse(code)
        features = GetFeatures(code)
        features.visit(tree)
        Features = np.array([features.Functions,
                             features.ClassDocsLen,
                             features.ClassNamesLen,
                             features.FunctionDocsLen,
                             features.FunctionNamesLen,
                             features.MaxDepth,
                             features.MeanDensity])
        X.append(Features)

X, y = np.array(X), np.array(y)
X = X.reshape(y.shape[0], 7)

scaler = StandardScaler()
X = scaler.fit_transform(X, y)

model = XGBClassifier(n_estimators=15, max_depth=7, random_state=42)
model.fit(X, y)

pickle.dump(model, open(dirs.model, "wb"))