# gui.py
import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('iris_classifier.joblib')
scaler = joblib.load('scaler.joblib')

# Map numerical labels to species names
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def classify_iris():
    try:
        # Get user inputs
        features = [
            float(entry_sepal_length.get()),
            float(entry_sepal_width.get()),
            float(entry_petal_length.get()),
            float(entry_petal_width.get())
        ]
        
        # Scale features and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        result = species[prediction[0]]
        
        # Display result
        label_result.config(text=f"Predicted Species: {result}", fg="green")
    except:
        label_result.config(text="Invalid input!", fg="red")

# Create GUI window
window = tk.Tk()
window.title("Iris Flower Classifier")
window.geometry("400x300")

# Input fields
ttk.Label(window, text="Sepal Length (cm):").pack(pady=5)
entry_sepal_length = ttk.Entry(window)
entry_sepal_length.pack()

ttk.Label(window, text="Sepal Width (cm):").pack(pady=5)
entry_sepal_width = ttk.Entry(window)
entry_sepal_width.pack()

ttk.Label(window, text="Petal Length (cm):").pack(pady=5)
entry_petal_length = ttk.Entry(window)
entry_petal_length.pack()

ttk.Label(window, text="Petal Width (cm):").pack(pady=5)
entry_petal_width = ttk.Entry(window)
entry_petal_width.pack()

# Classify button
ttk.Button(window, text="Classify", command=classify_iris).pack(pady=10)

# Result label (use tk.Label instead of ttk.Label to support fg)
label_result = tk.Label(window, text="", font=('Helvetica', 12))
label_result.pack()

window.mainloop()
