# AutoMileage AI — Streamlit application

A machine-learning web app that predicts vehicle fuel efficiency (MPG) using a classic Auto MPG dataset.

## Project files
- `app.py` — Streamlit UI, data cleaning, preprocessing pipelines, five regression algorithms, model evaluation, predictions and interactive charts.
- `requirements.txt` — required Python libraries.
- `81_dataset.csv` — supplied dataset.

## Run locally
Install Python 3.10 or newer, then place all three files in one folder.

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `81_dataset.csv` to the repository root.
3. Sign in at https://share.streamlit.io/ with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Deploy and test a prediction on the public app URL.

## Pipeline and models
- Validates the required dataset columns.
- Converts invalid numeric strings (including `?`) to missing values.
- Removes rows without the target `mpg`, and removes duplicates.
- Extracts a normalized brand token from `car name`.
- Imputes numeric features with their median, scales numeric features, imputes categorical features with the most frequent value, and one-hot encodes the brand.
- Uses scikit-learn `Pipeline` and `ColumnTransformer` to prevent preprocessing leakage during cross-validation.
- Compares Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and Support Vector Regression with 5-fold CV.
- Selects the lowest mean CV RMSE, reports a separate holdout evaluation, then fits the selected pipeline on all labeled rows for app predictions.

## Dataset warning
The supplied dataset is an automotive Auto MPG dataset, not a mental-health dataset. This app predicts fuel efficiency and does not assess mental well-being. It is based on older vehicles, so modern-car predictions may be unreliable.
