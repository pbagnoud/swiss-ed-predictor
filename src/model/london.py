# Imports
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import TimeSeriesSplit  # for future

import shap

import holidays

csv_path = "data/london_daily_temperature_aggregated_lag_traffic.csv"
out_path = "results/london_rf_traffic/"
os.makedirs(out_path, exist_ok=True)

FEATURE_COLS = [
    "temperature_min_c", "lag1", "weekday_num", "is_holidays", "traffic_tfl_tube_index"
]

TARGET_COL = "calls_assessed_by_clinician"

df = pd.read_csv(csv_path, parse_dates=["date"])
df["is_holidays"] = df.apply(lambda row: holidays.UK(subdiv="ENG", years=row["date"].year).get(row["date"]) is not None, axis=1)
df["weekday"] = df["date"].dt.day_name()
df["weekday_num"] = df["date"].dt.weekday

df = df.sort_values("date").reset_index(drop=True)
df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)

#df["date"] = pd.to_datetime(df["date"])


n = len(df)
train_end = int(n * 0.80)

train = df.iloc[:train_end]
val   = df.iloc[train_end:]

X_train = train[FEATURE_COLS].reset_index(drop=True)
y_train   = train[TARGET_COL].reset_index(drop=True)

X_val   = val[FEATURE_COLS].reset_index(drop=True)
y_val     = val[TARGET_COL].reset_index(drop=True)


# ── Preprocessing pipeline ─────────────────────────────────────────────────
# Identify column types (adjust manually if needed)
num_features = X_train.select_dtypes(include=np.number).columns.tolist()
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print('Numeric features :', num_features)
print('Categorical features:', cat_features)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# categorical_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# ])

categorical_transformer_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])  

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer_rf, cat_features)
])

# Training and evaluation
results = {}  # in case we want to run other models, or do cv, etc.

rf = Pipeline([
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=200, random_state=10))
])

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
results["rf"] = {
        'mse': mean_squared_error(y_val, y_pred_rf),
        'mae': mean_absolute_error(y_val, y_pred_rf),
        'r2': r2_score(y_val, y_pred_rf)
    }
pd.DataFrame([results["rf"]]).to_csv(os.path.join(out_path, "metrics.csv"), index=False)

print(results["rf"])
df = pd.DataFrame({"actual": y_val, "predicted": y_pred_rf})
df.to_csv(os.path.join(out_path, "results.csv"), index=False)

fig, ax = plt.subplots()

# scatter: actual vs predicted; perfect fit line (y = x)
ax.scatter(y_val, y_pred_rf, alpha=0.4, label='predictions')
min_val = min(y_val.min(), y_pred_rf.min())
max_val = max(y_val.max(), y_pred_rf.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='perfect fit')

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Predicted vs Actual")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_path, "actual_v_predicted.png"))
plt.close()

# residual plot
residuals = y_val - y_pred_rf
fig, ax = plt.subplots()
ax.scatter(y_pred_rf, residuals, alpha=0.4)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(out_path, "residuals.png"))
plt.close()



# Create SHAP TreeExplainer
explainer = shap.Explainer(rf)

shap_values = explainer.shap_values(X_val)

# Summary plot
shap.summary_plot(shap_values, X_val)

# Force plot for a single prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_val.iloc[0],
    matplotlib=True
)