# Import necessary libraries
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('process_data.csv')

# Separate features and target
X = df.drop('price', axis=1)
y_reg = df['price']

# Create price categories for classification
price_bins = pd.qcut(df['price'], q=3, labels=['Low', 'Medium', 'High'])
y_cls = price_bins

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Save feature columns for GUI compatibility
joblib.dump(X_encoded.columns, 'model_columns.pkl')

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_encoded, y_reg, test_size=0.2, random_state=42)

# Split data for classification (same X, different y)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_encoded, y_cls, test_size=0.2, random_state=42)

# Train regression model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
joblib.dump(reg_model, 'vehicle_price_model.pkl')

# Train classification model
cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
cls_model.fit(X_train_cls, y_train_cls)
joblib.dump(cls_model, 'vehicle_price_classifier.pkl')

# Print confirmation messages
print("✅ Models saved: vehicle_price_model.pkl, vehicle_price_classifier.pkl")
print("✅ Columns saved: model_columns.pkl")

# Evaluate regression
y_pred_reg = reg_model.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Regression Mean Squared Error: {mse:.2f}")

# Evaluate classification
y_pred_cls = cls_model.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"Classification Accuracy: {accuracy*100:.2f}%")
