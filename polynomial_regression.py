import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

ROLL_NO = "BT2024113"

MAX_DEGREE_VAR1 = 10
MAX_DEGREE_VAR2 = 20

ALPHA_VALUES = 10.0 ** np.arange(-3, 4)

N_SPLITS = 5

CV = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)


def create_pipeline():

    return Pipeline([
        (
            "polynomial",
            PolynomialFeatures(
                include_bias=False
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "regressor",
            Ridge(solver="lsqr")
        )
    ])

def select_model(
    X_train,
    y_train,
    maximum_degree,
    variable_name
):

    print("\n" + "=" * 70)
    print(f"MODEL SELECTION FOR {variable_name}")
    print("=" * 70)

    model = create_pipeline()

  
    parameter_grid = [

        # Ridge
        {
            "polynomial__degree":
                range(1, maximum_degree + 1),

            "regressor":
                [Ridge(solver="lsqr")],

            "regressor__alpha":
                ALPHA_VALUES
        },

        # Lasso
        {
            "polynomial__degree":
                range(1, maximum_degree + 1),

            "regressor":
                [
                    Lasso(
                        max_iter=50000,
                        tol=1e-4
                    )
                ],

            "regressor__alpha":
                ALPHA_VALUES
        }
    ]

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=CV,
        n_jobs=-1,
        refit=True,
        pre_dispatch="2*n_jobs"
    )

    print("Searching over polynomial degree, Ridge/Lasso, and alpha...")
    print("This can take some time, especially for high-degree Lasso models.")

    search.fit(
        X_train,
        y_train
    )

    best_degree = int(
        search.best_params_["polynomial__degree"]
    )

    best_alpha = float(
        search.best_params_["regressor__alpha"]
    )

    best_regressor = search.best_params_["regressor"]

    if isinstance(best_regressor, Ridge):
        best_method = "Ridge"
    else:
        best_method = "Lasso"

    best_cv_mse = -search.best_score_

    print("\nSearch completed.")

    print(
        f"Best method : {best_method}"
    )

    print(
        f"Best degree : {best_degree}"
    )

    print(
        f"Best alpha  : {best_alpha:g}"
    )

    print(
        f"Best CV MSE : {best_cv_mse:.8f}"
    )

    return (
        search.best_estimator_,
        best_method,
        best_degree,
        best_alpha,
        best_cv_mse
    )



def train_and_predict(
    train_file,
    test_file,
    output_file,
    maximum_degree,
    variable_name
):


    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

 
    feature_columns = [
        column
        for column in train_df.columns
        if column != "y"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["y"]

    X_test = test_df[feature_columns]


    (
        best_model,
        best_method,
        best_degree,
        best_alpha,
        best_cv_mse
    ) = select_model(
        X_train,
        y_train,
        maximum_degree,
        variable_name
    )

    train_prediction = best_model.predict(
        X_train
    )

    train_mse = mean_squared_error(
        y_train,
        train_prediction
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

 
    number_of_terms = (
        best_model
        .named_steps["polynomial"]
        .n_output_features_
    )

    test_prediction = best_model.predict(
        X_test
    )

    submission = pd.DataFrame({
        "y": test_prediction
    })

    submission.to_csv(
        output_file,
        index=False
    )

    print("\n" + "-" * 70)
    print(f"FINAL RESULT : {variable_name}")
    print("-" * 70)

    print(
        f"Selected method : {best_method}"
    )

    print(
        f"Selected degree : {best_degree}"
    )

    print(
        f"Selected alpha  : {best_alpha:g}"
    )

    print(
        f"CV MSE          : {best_cv_mse:.8f}"
    )

    print(
        f"Training MSE    : {train_mse:.8f}"
    )

    print(
        f"Training R2     : {train_r2:.8f}"
    )

    print(
        f"Polynomial terms: {number_of_terms}"
    )

    print(
        f"Predictions     : {len(test_prediction)}"
    )

    print(
        f"Saved file      : {output_file}"
    )

    return {
        "variable": variable_name,
        "method": best_method,
        "degree": best_degree,
        "alpha": best_alpha,
        "cv_mse": best_cv_mse,
        "train_mse": train_mse,
        "train_r2": train_r2
    }

def main():
    result_var1 = train_and_predict(
        train_file=f"{ROLL_NO}_train_var1.csv",
        test_file=f"{ROLL_NO}_test_var1.csv",
        output_file=f"{ROLL_NO}_pred_var1.csv",
        maximum_degree=MAX_DEGREE_VAR1,
        variable_name="var1"
    )


    result_var2 = train_and_predict(
        train_file=f"{ROLL_NO}_train_var2.csv",
        test_file=f"{ROLL_NO}_test_var2.csv",
        output_file=f"{ROLL_NO}_pred_var2.csv",
        maximum_degree=MAX_DEGREE_VAR2,
        variable_name="var2"
    )


    print("\n")
    print("=" * 70)
    print("FINAL AUTOMATIC MODEL SELECTION")
    print("=" * 70)

    print(
        f"var1 -> method = {result_var1['method']}, "
        f"degree = {result_var1['degree']}, "
        f"alpha = {result_var1['alpha']:g}, "
        f"CV MSE = {result_var1['cv_mse']:.8f}"
    )

    print(
        f"var2 -> method = {result_var2['method']}, "
        f"degree = {result_var2['degree']}, "
        f"alpha = {result_var2['alpha']:g}, "
        f"CV MSE = {result_var2['cv_mse']:.8f}"
    )

    print("\nPrediction files created:")

    print(
        f"{ROLL_NO}_pred_var1.csv"
    )

    print(
        f"{ROLL_NO}_pred_var2.csv"
    )

if __name__ == "__main__":
    main()
