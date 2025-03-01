# advertising_model_cpu.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # CPU version
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import joblib
import seaborn as sns

def load_data():
    df = pd.read_csv('Advertising.csv')
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    return df

def perform_eda(df):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    
    plt.subplot(1, 2, 2)
    sns.pairplot(df, diag_kind='kde')
    plt.suptitle("Data Distribution and Relationships", y=1.02)
    
    plt.tight_layout()
    plt.show()

class SalesPredictorApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Advertising Sales Predictor")
        self.window.geometry("400x350")
        self.create_widgets()
    
    def create_widgets(self):
        ttk.Label(self.window, text="TV Budget ($):").pack(pady=5)
        self.tv_entry = ttk.Entry(self.window)
        self.tv_entry.pack()
        
        ttk.Label(self.window, text="Radio Budget ($):").pack(pady=5)
        self.radio_entry = ttk.Entry(self.window)
        self.radio_entry.pack()
        
        ttk.Label(self.window, text="Newspaper Budget ($):").pack(pady=5)
        self.news_entry = ttk.Entry(self.window)
        self.news_entry.pack()
        
        ttk.Button(self.window, text="Predict Sales", command=self.predict_sales).pack(pady=10)
        self.result_label = ttk.Label(self.window, text="", font=('Helvetica', 12, 'bold'))
        self.result_label.pack()
        self.metrics_label = ttk.Label(self.window, text="", font=('Helvetica', 10))
        self.metrics_label.pack()
    
    def predict_sales(self):
        try:
            tv = float(self.tv_entry.get())
            radio = float(self.radio_entry.get())
            news = float(self.news_entry.get())
            prediction = self.model.predict(pd.DataFrame([[tv, radio, news]], 
                                           columns=['TV', 'Radio', 'Newspaper']))
            self.result_label.config(text=f"Predicted Sales: ${prediction[0]:.2f}", foreground="green")
        except:
            self.result_label.config(text="Invalid Input! Please enter numbers.", foreground="red")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    df = load_data()
    perform_eda(df)
    
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'R²': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    print("\nModel Performance:")
    print(f"R² Score: {metrics['R²']:.4f}")
    print(f"Mean Squared Error: {metrics['MSE']:.2f}")
    print(f"Mean Absolute Error: {metrics['MAE']:.2f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales Performance")
    plt.show()
    
    joblib.dump(model, "advertising_model_cpu.pkl")
    app = SalesPredictorApp(model)
    app.run()
