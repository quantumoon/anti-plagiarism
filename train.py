import ast
import argparse
import os
import pickle
import numpy as np

from xgboost import XGBClassifier


class GetFeatures(ast.NodeVisitor):
    """
    Methods of this class allow to get additional features from the code to create a feature object matrix.

    Features : number of functions, total length of all function documentations (docs),
               total length of all class docs, total length of all class names,
               total length of all function names, max depth of conditions & cycles in the code
               and average density
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

    model = XGBClassifier(n_estimators=15, max_depth=7, random_state=42)
    model.fit(X, y)

    pickle.dump(model, open(dirs.model, "wb"))


if __name__ == "__main__":
    main()
