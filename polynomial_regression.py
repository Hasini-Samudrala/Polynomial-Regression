import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score


# ============================================================
# 1. LOAD DATA
# ============================================================

train1 = pd.read_csv("BT2024113_train_var1.csv")
test1 = pd.read_csv("BT2024113_test_var1.csv")

train2 = pd.read_csv("BT2024113_train_var2.csv")
test2 = pd.read_csv("BT2024113_test_var2.csv")


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X1 = train1.drop(columns="y")
y1 = train1["y"]

X2 = train2.drop(columns="y")
y2 = train2["y"]


# ============================================================
# 3. CROSS-VALIDATION SETUP
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. FIND BEST DEGREE FOR VAR1
# ============================================================

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


# ============================================================
# 5. FIND BEST DEGREE FOR VAR2
# ============================================================

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


# ============================================================
# 6. CREATE FINAL VAR1 MODEL
# ============================================================

model1 = make_pipeline(
    PolynomialFeatures(
        degree=best_degree_var1,
        include_bias=False
    ),
    StandardScaler(),
    Ridge(alpha=30)
)


# ============================================================
# 7. CREATE FINAL VAR2 MODEL
# ============================================================

model2 = make_pipeline(
    PolynomialFeatures(
        degree=best_degree_var2,
        include_bias=False
    ),
    StandardScaler(),
    Ridge(alpha=2)
)


# ============================================================
# 8. TRAIN FINAL MODELS
# ============================================================

model1.fit(X1, y1)
model2.fit(X2, y2)


# ============================================================
# 9. GENERATE TEST PREDICTIONS
# ============================================================

pred1 = model1.predict(test1)
pred2 = model2.predict(test2)


# ============================================================
# 10. SAVE PREDICTION FILES
# ============================================================

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


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

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