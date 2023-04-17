import argparse
import ast
import astor
import pickle
import numpy as np

from train import GetFeatures


class LevenshteinDist():
    """
    Class for preprocessing Python codes and finding the Levenshtein distance between them
    """


    def _remove_docs(self, parsed):
        """
        Separates all documentations (docs) in the code from the code itself

        Parameters:
            parsed (ast.AST) : an abstract syntax tree of your code

        Returns:
            clear_code (str) : your initial code without docs
            docs (list) : a list of docs in your code
        """
        docs = []
        for node in ast.walk(parsed):
            if isinstance(node, (ast.FunctionDef,
                                 ast.ClassDef,
                                 ast.AsyncFunctionDef,
                                 ast.Module)):
                doc = ast.get_docstring(node)
                if doc is None:
                    docs.append("")
                else:
                    docs.append(doc)
            else:
                continue

            if not len(node.body):
                continue

            if not isinstance(node.body[0], ast.Expr):
                continue

            if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
                continue

            node.body = node.body[1:]

        return astor.to_source(parsed), docs


    def _get_code(self, path):
        """
        Processes two Python codes

        Parameters:
            path (str) : a path to your .py file

        Returns:
            sorted_clear_code (str) : your code without docs sorted by rows
            sorted_docs (str) : docs sorted by rows
        """

        clear_code = []
        with open(path, "r", encoding="utf-8") as code_file:
            code, docs = self._remove_docs(ast.parse(code_file.read()))

        for line in code.split("\n"):
            if not line.strip():
                continue
            clear_code.append(line.strip())

        return "".join(sorted(clear_code)), "".join(sorted(docs))


    def dist(self, text_1, text_2):
        """
        Finds the Levenshtein distance between two texts using the Wagner — Fischer algorithm

        Parameters:
            text_1 (str) : the first text
            text_2 (str) : the second text

        Returns:
            distance (int) : the Levenshtein distance itself
        """

        n, m = min(len(text_1), len(text_2)), max(len(text_1), len(text_2))
        if len(text_1) > len(text_2):
            text_1, text_2 = text_2, text_1

        now = range(n + 1)

        for i in range(1, m + 1):
            prev, now = now, [i] + [0] * n
            for j in range(1, n + 1):
                case_1 = prev[j] + 1
                case_2 = now[j - 1] + 1
                case_3 = prev[j - 1] + (text_1[j - 1] != text_2[i - 1])

                now[j] = min(case_1, case_2, case_3)

        return int(now[n])


    def _get_similarity(self, codes_pair, weight_for_docs=0.25):
        """
        Finds how similar two Python codes are (using the Levenshtein distance)

        Parameters:
            codes_pair (str) : row with pathes to .py files separated by space

        Returns:
            similarity (float) : a number from 0 to 1 which symbolizes how similar codes are
            (here 1 - absolutely similar; 0 - absolutely different)
        """

        path1, path2 = map(str, codes_pair.split())
        code1, docs1 = self._get_code(path1)
        code2, docs2 = self._get_code(path2)

        return 1 - self.dist(code1, code2) / max(len(code1), len(code2)) - weight_for_docs * self.dist(docs1, docs2) / max(len(docs1), len(docs2))


def get_txts():
    """
    Allows user to run this script from the console terminal
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--model", type=str)
    
    return parser.parse_args()


def get_prediction(codes_pair, clf):
    """
    Finds how similar two Python codes are (using ML methods)

    Parameters:
        codes_pair (str) : row with pathes to .py files separated by space
        clf : fitted classification ML model

    Returns:
        similarity (float) : a number from 0 to 1 which symbolizes how similar codes are
    """

    path1, path2 = map(str, codes_pair.split())
    pair = [open(path1, "r", encoding="utf-8").read(),
             open(path2, "r", encoding="utf-8").read()]

    X_test = []
    for code in pair:
        analyzer = GetFeatures(code)
        analyzer.visit(ast.parse(code))
        features = analyzer._get_features()
        X_test.append(features)
    X_test = np.array(X_test).reshape(2, 7)
    preds = clf.predict_proba(X_test)
    
    return preds[0][0] * preds[1][1] + preds[0][1] * preds[1][0]


def main():
    dirs = get_txts()
    model = pickle.load(open(dirs.model, "rb"))

    with open(dirs.input, "r") as f:
        scores = open(dirs.output, "w")
        for line in f:
            dist = LevenshteinDist()._get_similarity(line)
            pred = get_prediction(line, model)
            score = round((dist + pred) / 2, 3)
            scores.write(str(score) + "\n")
        scores.close()


if __name__ == "__main__":
    main()