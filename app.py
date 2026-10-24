import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

st.set_page_config(page_title="AutoMileage AI", page_icon="🚗", layout="wide")

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "81_dataset.csv"
TARGET = "mpg"
NUMERIC_FEATURES = [
    "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model year", "origin",
]
CATEGORICAL_FEATURES = ["car_brand"]
MODEL_NAMES = ["Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting", "Support Vector Regression"]


def extract_brand(name):
    """Extract a normalized first-token car brand from the vehicle name."""
    if pd.isna(name):
        return "unknown"
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", str(name).strip().lower())
    tokens = cleaned.split()
    if not tokens:
        return "unknown"
    brand = tokens[0]
    aliases = {
        "vw": "volkswagen",
        "vokswagen": "volkswagen",
        "chevy": "chevrolet",
        "maxda": "mazda",
        "toyouta": "toyota",
        "mercedes": "mercedes-benz",
    }
    return aliases.get(brand, brand)


def clean_dataset(raw):
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = set(NUMERIC_FEATURES + ["car name", TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(
            df[col].replace(["?", "NA", "N/A", "na", ""], np.nan),
            errors="coerce",
        )
    df["car name"] = df["car name"].fillna("Unknown car").astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET]).drop_duplicates().copy()
    df["car_brand"] = df["car name"].apply(extract_brand)
    return df


def make_pipeline(estimator):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessing", preprocessor), ("model", estimator)])


def candidates():
    return {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=3, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=2, random_state=42
        ),
        "Support Vector Regression": SVR(C=10, epsilon=0.2, kernel="rbf"),
    }


@st.cache_data
def load_and_clean(uploaded_bytes=None):
    if uploaded_bytes is not None:
        from io import BytesIO
        raw = pd.read_csv(BytesIO(uploaded_bytes))
    else:
        if not DEFAULT_DATA.exists():
            raise FileNotFoundError(
                "81_dataset.csv not found. Place it beside app.py or upload the CSV above."
            )
        raw = pd.read_csv(DEFAULT_DATA)
    return clean_dataset(raw)


@st.cache_resource
def train_and_evaluate(data_json):
    # Rebuild from cached JSON so Streamlit cache keys remain stable and explicit.
    from io import StringIO
    df = pd.read_json(StringIO(data_json), orient="split")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    estimators = candidates()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, estimator in estimators.items():
        scores = cross_validate(
            make_pipeline(estimator), X, y, cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
            error_score="raise",
        )
        rows.append({
            "Algorithm": name,
            "CV RMSE (MPG)": -scores["test_rmse"].mean(),
            "CV MAE (MPG)": -scores["test_mae"].mean(),
            "CV R²": scores["test_r2"].mean(),
        })

    leaderboard = pd.DataFrame(rows).sort_values("CV RMSE (MPG)").reset_index(drop=True)
    best_name = leaderboard.iloc[0]["Algorithm"]

    # Independent holdout sanity check; do not use holdout scores to choose the model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    holdout = make_pipeline(estimators[best_name])
    holdout.fit(X_train, y_train)
    predicted = holdout.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "R²": float(r2_score(y_test, predicted)),
    }

    # Refit selected algorithm on all available labeled data for production predictions.
    final_model = make_pipeline(estimators[best_name])
    final_model.fit(X, y)
    return final_model, best_name, leaderboard, metrics


def main():
    st.title("🚗 AutoMileage AI")
    st.write("A machine-learning dashboard for classic-car fuel-efficiency prediction.")
    st.caption("Target: MPG (miles per gallon) • Five algorithms • Automatic model selection")

    st.info(
        "Dataset notice: the provided CSV is the Auto MPG automotive dataset, not a mental "
        "well-being dataset. This application predicts fuel efficiency and does not assess health."
    )

    with st.sidebar:
        st.header("Data source")
        uploaded = st.file_uploader("Upload a compatible CSV (optional)", type=["csv"])
        st.markdown("---")
        st.write("**Expected columns**")
        st.code("mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin, car name")
        st.caption("If no file is uploaded, the app loads 81_dataset.csv beside app.py.")

    try:
        df = load_and_clean(uploaded.getvalue() if uploaded else None)
        fingerprint_json = df.to_json(orient="split", date_format="iso")
        model, best_name, leaderboard, metrics = train_and_evaluate(fingerprint_json)
    except Exception as exc:
        st.error(f"Could not prepare the data or train the models: {exc}")
        st.stop()

    missing_horsepower = int(df["horsepower"].isna().sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Usable records", f"{len(df):,}")
    m2.metric("Missing horsepower", missing_horsepower)
    m3.metric("Best cross-validation model", best_name)
    m4.metric("Holdout RMSE", f'{metrics["RMSE"]:.2f} MPG')

    tabs = st.tabs(["Predict", "Explore data", "Model comparison", "About"])

    with tabs[0]:
        st.subheader("Predict a car's fuel efficiency")
        car_names = sorted(df["car name"].dropna().astype(str).unique().tolist())
        selected_name = st.selectbox("Choose a dataset car or select custom input", ["Custom car"] + car_names)
        row = None
        if selected_name != "Custom car":
            matches = df[df["car name"].astype(str) == selected_name]
            if not matches.empty:
                row = matches.iloc[0]

        def val(col, default):
            if row is not None and pd.notna(row[col]):
                return float(row[col])
            return default

        with st.form("prediction_form"):
            car_name = st.text_input(
                "Car name",
                value=selected_name if selected_name != "Custom car" else "",
                placeholder="e.g. Toyota Corolla",
            )
            a, b, c = st.columns(3)
            with a:
                cylinders_choices = [3, 4, 5, 6, 8]
                default_cyl = int(val("cylinders", 4))
                cylinders = st.selectbox(
                    "Cylinders", cylinders_choices,
                    index=cylinders_choices.index(default_cyl) if default_cyl in cylinders_choices else 1,
                )
                displacement = st.number_input("Displacement", 40.0, 800.0, float(val("displacement", 140)), step=1.0)
                horsepower = st.number_input("Horsepower", 20.0, 500.0, float(val("horsepower", 95)), step=1.0)
            with b:
                weight = st.number_input("Weight (lbs)", 1000, 7000, int(val("weight", 2800)), step=50)
                acceleration = st.number_input("Acceleration", 5.0, 30.0, float(val("acceleration", 15)), step=0.5)
                model_year = st.slider("Model year (dataset codes 70–82)", 70, 82, int(val("model year", 80)))
            with c:
                origin_map = {1: "USA", 2: "Europe", 3: "Japan"}
                default_origin = int(val("origin", 1))
                origin = st.selectbox(
                    "Origin", [1, 2, 3],
                    index=[1, 2, 3].index(default_origin) if default_origin in [1, 2, 3] else 0,
                    format_func=lambda n: f"{n} — {origin_map[n]}",
                )
                st.write("")
                st.caption("Values should be within the range of the training dataset.")
                submit = st.form_submit_button("Predict mileage", type="primary", use_container_width=True)

        if submit:
            if not car_name.strip():
                st.warning("Enter a car name to continue.")
            else:
                sample = pd.DataFrame([{
                    "cylinders": cylinders,
                    "displacement": displacement,
                    "horsepower": horsepower,
                    "weight": weight,
                    "acceleration": acceleration,
                    "model year": model_year,
                    "origin": origin,
                    "car_brand": extract_brand(car_name),
                }])
                mpg = float(model.predict(sample)[0])
                km_per_litre = mpg * 0.425143707
                p1, p2 = st.columns(2)
                p1.metric("Estimated mileage", f"{mpg:.2f} MPG")
                p2.metric("Approx. metric mileage", f"{km_per_litre:.2f} km/L")
                st.success(f"Estimated fuel efficiency for {car_name.strip()}: {mpg:.2f} MPG.")
                st.caption(
                    f"Selected model: {best_name}. This is a model estimate, not an official "
                    "manufacturer rating. Modern cars may be outside this dataset's training range."
                )

    with tabs[1]:
        st.subheader("Explore the dataset")
        left, right = st.columns(2)
        with left:
            fig = px.histogram(df, x="mpg", nbins=25, title="Fuel-efficiency distribution (MPG)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(
                df, x="weight", y="mpg", color="origin",
                hover_data=["car name", "horsepower", "model year"],
                title="Vehicle weight vs MPG",
                labels={"weight": "Weight (lbs)", "mpg": "Miles per gallon"},
            )
            st.plotly_chart(fig, use_container_width=True)
        left, right = st.columns(2)
        with left:
            by_year = df.groupby("model year", as_index=False)["mpg"].mean()
            fig = px.line(by_year, x="model year", y="mpg", markers=True, title="Average MPG by model year")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(
                df, x="horsepower", y="mpg", color="cylinders",
                hover_data=["car name", "weight"], title="Horsepower vs MPG",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Cleaned data preview")
        st.dataframe(df.head(30), use_container_width=True)

    with tabs[2]:
        st.subheader("Five-algorithm leaderboard")
        st.write("The winning algorithm is selected using the lowest mean 5-fold cross-validation RMSE.")
        display = leaderboard.copy()
        display[["CV RMSE (MPG)", "CV MAE (MPG)", "CV R²"]] = display[
            ["CV RMSE (MPG)", "CV MAE (MPG)", "CV R²"]
        ].round(3)
        st.dataframe(display, hide_index=True, use_container_width=True)
        fig = px.bar(
            leaderboard, x="Algorithm", y="CV RMSE (MPG)",
            title="Cross-validation RMSE (lower is better)",
        )
        st.plotly_chart(fig, use_container_width=True)
        q1, q2, q3 = st.columns(3)
        q1.metric("Holdout MAE", f'{metrics["MAE"]:.2f} MPG')
        q2.metric("Holdout RMSE", f'{metrics["RMSE"]:.2f} MPG')
        q3.metric("Holdout R²", f'{metrics["R²"]:.3f}')
        st.caption("Holdout metrics are a separate 20% test split. They are estimates and may vary with data.")

    with tabs[3]:
        st.subheader("How AutoMileage AI works")
        st.markdown("""
        1. **Extract:** reads the supplied CSV or an uploaded compatible CSV.
        2. **Clean:** converts invalid numeric values such as `?` to missing values, removes rows without MPG, and removes duplicates.
        3. **Engineer features:** extracts a normalized car-brand feature from `car name`.
        4. **Preprocess:** median-imputes numeric values, scales numeric features, imputes categorical values, and one-hot encodes car brands inside an sklearn pipeline.
        5. **Compare:** evaluates Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and Support Vector Regression using 5-fold cross-validation.
        6. **Select:** chooses the lowest cross-validation RMSE model, reports a separate holdout evaluation, then refits the chosen pipeline on all labeled data.
        7. **Predict:** provides MPG and approximate km/L estimates.

        **Important:** this is an educational model trained on an older, relatively small dataset. It should not be treated as an official fuel-economy rating.
        """)


if __name__ == "__main__":
    main()
