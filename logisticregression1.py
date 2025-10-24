import numpy as np
import pandas as pd

# load
df = pd.read_csv("titanic.csv")
df = df[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].copy()

# cleaning
for c in ["Age","Fare"]:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())
if df["Embarked"].isna().any():
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
#simple features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["Sex"] = (df["Sex"] == "male").astype(int)  # male=1, female=0
emb = pd.get_dummies(df["Embarked"], prefix="Emb", drop_first=True)
X_num = df[["Pclass","Age","SibSp","Parch","Fare","FamilySize","Sex"]].copy()

# scaling
for c in ["Pclass","Age","SibSp","Parch","Fare","FamilySize"]:
    m = X_num[c].mean(); s = X_num[c].std(ddof=0)
    X_num[c] = (X_num[c]-m)/(s if s>0 else 1.0)

X_df = pd.concat([X_num, emb], axis=1)
X = X_df.values.astype(float)
y = df["Survived"].astype(int).values

# stratified 80/20 split
rng = np.random.RandomState(42)
i0 = np.where(y==0)[0]; rng.shuffle(i0)
i1 = np.where(y==1)[0]; rng.shuffle(i1)
n0t = int(0.2*len(i0)); n1t = int(0.2*len(i1))
test_idx = np.concatenate([i0[:n0t], i1[:n1t]])
train_idx = np.concatenate([i0[n0t:], i1[n1t:]])
rng.shuffle(test_idx); rng.shuffle(train_idx)
X_tr, y_tr = X[train_idx], y[train_idx]
X_te, y_te = X[test_idx],  y[test_idx]

#logistic regression
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0/(1.0+np.exp(-z))

def train_logreg(X, y, lr=0.1, n_iter=2000, fit_intercept=True):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        z = X@w + (b if fit_intercept else 0.0)
        p = sigmoid(z)
        err = p - y
        grad_w = (X.T@err)/n
        w -= lr*grad_w
        if fit_intercept:
            b -= lr*err.mean()
    return w, b

def predict_proba(X, w, b, fit_intercept=True):
    return sigmoid(X@w + (b if fit_intercept else 0.0))

def predict(X, w, b, thr=0.5, fit_intercept=True):
    return (predict_proba(X, w, b, fit_intercept)>=thr).astype(int)

# train
w, b = train_logreg(X_tr, y_tr, lr=0.15, n_iter=2000, fit_intercept=True)

#metrics
def accuracy(y_true, y_pred):
    return float((y_true==y_pred).mean())

def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

def confusion_matrix(y_true, y_pred):
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    tp = np.sum((y_true==1)&(y_pred==1))
    return np.array([[tn, fp],[fn, tp]])

# evaluate
y_hat = predict(X_te, w, b, thr=0.5, fit_intercept=True)
acc = accuracy(y_te, y_hat)
prec, rec, f1 = precision_recall_f1(y_te, y_hat)
cm = confusion_matrix(y_te, y_hat)

print("Validation results")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion matrix [[tn, fp],[fn, tp]]:\n", cm)