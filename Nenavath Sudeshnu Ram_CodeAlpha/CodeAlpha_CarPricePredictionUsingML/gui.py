# car_gui.py
import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('car_price_predictor.joblib')

def predict_price():
    try:
        # Get user inputs
        input_data = {
            'Brand': brand_var.get(),
            'Year': int(year_entry.get()),
            'Present_Price': float(present_price_entry.get()),
            'Driven_kms': int(driven_kms_entry.get()),
            'Fuel_Type': fuel_var.get(),
            'Selling_type': selling_type_var.get(),
            'Transmission': transmission_var.get(),
            'Owner': int(owner_entry.get())
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict
        prediction = model.predict(input_df)[0]
        result_label.config(text=f"Predicted Selling Price: ₹{prediction:.2f} Lakh", fg="green")
    except:
        result_label.config(text="Invalid Input!", fg="red")

# Create GUI window
window = tk.Tk()
window.title("Car Price Predictor")
window.geometry("600x500")

# Input fields
ttk.Label(window, text="Brand:").pack(pady=5)
brand_var = tk.StringVar()
brand_dropdown = ttk.Combobox(window, textvariable=brand_var, values=["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Renault", "Mahindra"])
brand_dropdown.pack()

ttk.Label(window, text="Year:").pack(pady=5)
year_entry = ttk.Entry(window)
year_entry.pack()

ttk.Label(window, text="Present Price (₹ Lakh):").pack(pady=5)
present_price_entry = ttk.Entry(window)
present_price_entry.pack()

ttk.Label(window, text="Driven Kilometers:").pack(pady=5)
driven_kms_entry = ttk.Entry(window)
driven_kms_entry.pack()

ttk.Label(window, text="Fuel Type:").pack(pady=5)
fuel_var = tk.StringVar()
fuel_dropdown = ttk.Combobox(window, textvariable=fuel_var, values=["Petrol", "Diesel", "CNG"])
fuel_dropdown.pack()

ttk.Label(window, text="Selling Type:").pack(pady=5)
selling_type_var = tk.StringVar()
selling_type_dropdown = ttk.Combobox(window, textvariable=selling_type_var, values=["Dealer", "Individual"])
selling_type_dropdown.pack()

ttk.Label(window, text="Transmission:").pack(pady=5)
transmission_var = tk.StringVar()
transmission_dropdown = ttk.Combobox(window, textvariable=transmission_var, values=["Manual", "Automatic"])
transmission_dropdown.pack()

ttk.Label(window, text="Number of Previous Owners:").pack(pady=5)
owner_entry = ttk.Entry(window)
owner_entry.pack()

# Predict button
ttk.Button(window, text="Predict Selling Price", command=predict_price).pack(pady=10)

# Result label
result_label = tk.Label(window, text="", font=('Helvetica', 12, 'bold'))
result_label.pack()

window.mainloop()
