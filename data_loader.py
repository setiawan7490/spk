import pandas as pd

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "Label_Asli"
    ]

    df = pd.read_csv(url, header=None, names=columns)

    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    df["Label_Asli"] = df["Label_Asli"].map(label_map)

    target_names = ["Setosa", "Versicolor", "Virginica"]

    return df, target_names
