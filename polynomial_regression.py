import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score


# loading data
train1 = pd.read_csv("BT2024113_train_var1.csv")
test1 = pd.read_csv("BT2024113_test_var1.csv")

train2 = pd.read_csv("BT2024113_train_var2.csv")
test2 = pd.read_csv("BT2024113_test_var2.csv")


# seperating features,target 
X1 = train1.drop(columns="y")
y1 = train1["y"]

X2 = train2.drop(columns="y")
y2 = train2["y"]


# cross validation setting up
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# finding the best degree for var1
print("Testing polynomial degrees for Var1...")

var1_results = []

for degree in range(1, 11):

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        StandardScaler(),
        Ridge(alpha=30)
    )

    scores = cross_val_score(
        model,
        X1,
        y1,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    mse = -scores.mean()

    var1_results.append((degree, mse))

    print(
        f"Var1 - Degree {degree}: "
        f"MSE = {mse:.6f}"
    )


best_degree_var1 = min(
    var1_results,
    key=lambda x: x[1]
)[0]

print(
    f"\nBest degree for Var1: "
    f"{best_degree_var1}"
)


#finding the best degree for var2
print("\nTesting polynomial degrees for Var2...")

var2_results = []

for degree in range(1, 21):

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        StandardScaler(),
        Ridge(alpha=2)
    )

    scores = cross_val_score(
        model,
        X2,
        y2,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    mse = -scores.mean()

    var2_results.append((degree, mse))

    print(
        f"Var2 - Degree {degree}: "
        f"MSE = {mse:.6f}"
    )


best_degree_var2 = min(
    var2_results,
    key=lambda x: x[1]
)[0]

print(
    f"\nBest degree for Var2: "
    f"{best_degree_var2}"
)


# creating final var1 model 
model1 = make_pipeline(
    PolynomialFeatures(
        degree=best_degree_var1,
        include_bias=False
    ),
    StandardScaler(),
    Ridge(alpha=30)
)


# creating var 2 model final
model2 = make_pipeline(
    PolynomialFeatures(
        degree=best_degree_var2,
        include_bias=False
    ),
    StandardScaler(),
    Ridge(alpha=2)
)


# training teh final models
model1.fit(X1, y1)
model2.fit(X2, y2)


# generating test predictions
pred1 = model1.predict(test1)
pred2 = model2.predict(test2)


# saving those files
pd.DataFrame({
    "y": pred1
}).to_csv(
    "BT2024113_pred_var1.csv",
    index=False
)

pd.DataFrame({
    "y": pred2
}).to_csv(
    "BT2024113_pred_var2.csv",
    index=False
)

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print("Var1 best degree:", best_degree_var1)
print("Var1 Ridge alpha:", 30)

print("Var2 best degree:", best_degree_var2)
print("Var2 Ridge alpha:", 2)

print("\nPrediction files created successfully:")
print("BT2024113_pred_var1.csv")
print("BT2024113_pred_var2.csv")
