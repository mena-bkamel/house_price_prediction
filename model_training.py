import joblib
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from data_preparation import load_and_prepare_data

def prepare_features(df):
    """Prepare features and target variable"""
    # Define which columns to use as features
    feature_cols = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
        'parking', 'prefarea', 'price_per_sqft', 'room_ratio',
        'furnish_furnished', 'furnish_semi-furnished', 'furnish_unfurnished'
    ]

    # Ensure we only use columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df["price"]

    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    return metrics

def train_models(X_train, y_train, X_test, y_test):
    """Train and compair multiple models"""
    os.makedirs("models", exist_ok=True)

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # Replaces missing values (NaNs) in numeric columns with the median of each column., Median is more robust to outliers than mean.
        ("scaler", StandardScaler()),
    ])

    # Apply the preprocessing using ColumnTransformer.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
    }
    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        joblib.dump(pipeline, f"models/{name.lower().replace(" ", "_")}_model.pkl")
        print(f"Save {name} model")

        print(f"{name} Performance:")
        for metric, value in metrics.items():
            print(f" {metric}: {value:.2f}")

    return results


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning on the best model"""
    print("\nPerforming hyperparameter tuning on Gradient Boosting...")

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10],
    }
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    print("Best Parameters found:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"models/tuned_gradient_boosting_model.pkl")

    return best_model



if __name__ == '__main__':
    df = load_and_prepare_data()
    if df is not None:
        X,y = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = train_models(X_train, y_train, X_test, y_test)
        best_model = hyperparameter_tuning(X_train, y_train)

        tuned_metrics = evaluate_model(best_model, X_test, y_test)
        print("\nTuned Gradient Boosting Performance:")
        for metric, value in tuned_metrics.items():
            print(f" {metric}: {value:.2f}")

