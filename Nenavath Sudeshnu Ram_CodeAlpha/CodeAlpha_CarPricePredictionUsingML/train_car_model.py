# train_car_model.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
# Load dataset
df = pd.read_csv("car data.csv")

# Extract Brand from Car_Name (e.g., "Maruti Swift Dzire VDI" -> "Maruti")
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])
df = df.drop('Car_Name', axis=1)  # Drop original column

# Define features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Categorical and numerical features
categorical_features = ['Brand', 'Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f} Lakh")
print(f"R² Score: {r2:.2f}")

# Save model
joblib.dump(model, 'car_price_predictor.joblib')

# Visualizations (Actual vs. Predicted and Residual Plots)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Selling Price (₹ Lakh)')
plt.ylabel('Predicted Selling Price (₹ Lakh)')
plt.title('Actual vs. Predicted Prices')
plt.savefig('actual_vs_predicted.png')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Selling Price (₹ Lakh)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.savefig('residual_plot.png')
plt.show()
