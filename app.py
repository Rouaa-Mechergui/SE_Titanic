import tkinter as tk
from tkinter import ttk, messagebox


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ======================
# 1) Charger la BD Titanic (CSV en ligne)
# ======================
CSV_URL = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
df = pd.read_csv(CSV_URL)
df.columns = df.columns.str.strip().str.lower()


TARGET = "survived"
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]


df2 = df[FEATURES + [TARGET]].copy()
X = df2[FEATURES]
y = df2[TARGET]


# ======================
# 2) Pré-traitement + arbre de décision
# ======================
num = ["age", "sibsp", "parch", "fare"]
cat = ["pclass", "sex", "embarked"]


preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
    ]
)


model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
model.fit(X_train, y_train)
import matplotlib.pyplot as plt
from sklearn import tree


clf = model.named_steps["clf"]  # l'arbre dans le Pipeline
plt.figure(figsize=(26, 12))
tree.plot_tree(clf, filled=True, rounded=True, class_names=["No", "Yes"])
plt.savefig("tree.png", dpi=200, bbox_inches="tight")
plt.show()


acc = accuracy_score(y_test, model.predict(X_test))


def diagnostiquer(pclass, sex, age, sibsp, parch, fare, embarked):
    row = pd.DataFrame([{
        "pclass": int(pclass),
        "sex": sex,
        "age": float(age),
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": float(fare),
        "embarked": embarked
    }])
    proba_survie = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])
    decision = "Survie probable" if pred == 1 else "Décès probable"
    return decision, proba_survie


# ======================
# 3) Interface Tkinter
# ======================
root = tk.Tk()
root.title("SE Titanic")


main = ttk.Frame(root, padding=12)
main.grid(row=0, column=0, sticky="nsew")


ttk.Label(main, text=f"Système expert Titanic — Prédiction de survie | Précision (test): {acc:.3f}").grid(
    row=0, column=0, columnspan=2, pady=(0, 10), sticky="w"
)


# Champs
pclass_var = tk.StringVar(value="3")
sex_var = tk.StringVar(value="male")
age_var = tk.StringVar(value="25")
sibsp_var = tk.StringVar(value="0")
parch_var = tk.StringVar(value="0")
fare_var = tk.StringVar(value="7.25")
embarked_var = tk.StringVar(value="S")


ttk.Label(main, text="Classe du billet (1=1ère, 2=2ème, 3=3ème)").grid(row=1, column=0, sticky="w")
ttk.Combobox(main, textvariable=pclass_var, values=["1", "2", "3"], state="readonly").grid(row=1, column=1, sticky="ew")


ttk.Label(main, text="Sexe (male/female)").grid(row=2, column=0, sticky="w")
ttk.Combobox(main, textvariable=sex_var, values=["male", "female"], state="readonly").grid(row=2, column=1, sticky="ew")


ttk.Label(main, text="Âge (en années)").grid(row=3, column=0, sticky="w")
ttk.Entry(main, textvariable=age_var).grid(row=3, column=1, sticky="ew")


ttk.Label(main, text="Frères/sœurs + conjoint à bord").grid(row=4, column=0, sticky="w")
ttk.Entry(main, textvariable=sibsp_var).grid(row=4, column=1, sticky="ew")


ttk.Label(main, text="Parents/enfants à bord").grid(row=5, column=0, sticky="w")
ttk.Entry(main, textvariable=parch_var).grid(row=5, column=1, sticky="ew")


ttk.Label(main, text="Prix du billet").grid(row=6, column=0, sticky="w")
ttk.Entry(main, textvariable=fare_var).grid(row=6, column=1, sticky="ew")


ttk.Label(main, text="Port d'embarquement: S=Southampton, C=Cherbourg, Q=Queenstown").grid(row=7, column=0, sticky="w")
ttk.Combobox(main, textvariable=embarked_var, values=["S", "C", "Q"], state="readonly").grid(row=7, column=1, sticky="ew")


result_var = tk.StringVar(value="Décision: ...")
ttk.Label(main, textvariable=result_var).grid(row=9, column=0, columnspan=2, pady=(10, 0), sticky="w")


def on_predict():
    try:
        decision, p = diagnostiquer(
            pclass_var.get(),
            sex_var.get(),
            age_var.get(),
            sibsp_var.get(),
            parch_var.get(),
            fare_var.get(),
            embarked_var.get(),
        )
        result_var.set(f"Décision: {decision} | p(survie)={p:.2f}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


ttk.Button(main, text="Diagnostiquer", command=on_predict).grid(row=8, column=0, columnspan=2, pady=10, sticky="ew")


main.columnconfigure(1, weight=1)
root.mainloop()