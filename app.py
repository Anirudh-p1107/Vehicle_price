import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import joblib

# Load models and columns
reg_model = joblib.load('vehicle_price_model.pkl')
cls_model = joblib.load('vehicle_price_classifier.pkl')
columns = joblib.load('model_columns.pkl')

# Load your dataset to get feature types
df = pd.read_csv('process_data.csv')
X = df.drop('price', axis=1)

class VehiclePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Price Predictor")
        self.root.geometry("800x800")
        ctk.set_appearance_mode("light")  # Change to 'light', 'dark', or 'system'
        ctk.set_default_color_theme("blue")  # Use 'blue', 'green', 'dark-blue'

        self.inputs = {}

        # Scrollable Frame
        scroll_frame = ctk.CTkScrollableFrame(root, corner_radius=15, width=750, height=650)
        scroll_frame.pack(padx=20, pady=20, fill="both", expand=True)

        title_label = ctk.CTkLabel(scroll_frame, text="Vehicle Price & Category Predictor", font=("Arial Black", 24))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Create input widgets in grid format
        for idx, col in enumerate(X.columns):
            label = ctk.CTkLabel(scroll_frame, text=col, font=("Arial", 16))
            label.grid(row=idx + 1, column=0, sticky="w", padx=20, pady=10)

            if X[col].dtype == 'object':
                combo = ctk.CTkComboBox(scroll_frame, values=sorted(X[col].astype(str).unique()), width=300)
                combo.set(sorted(X[col].astype(str).unique())[0])
                combo.grid(row=idx + 1, column=1, padx=20, pady=10)
                self.inputs[col] = combo
            else:
                if col.lower() in ['year', 'doors']:
                    spin = ctk.CTkEntry(scroll_frame, width=300)
                    spin.insert(0, str(int(X[col].mean())))
                else:
                    spin = ctk.CTkEntry(scroll_frame, width=300)
                    spin.insert(0, str(float(X[col].mean())))
                spin.grid(row=idx + 1, column=1, padx=20, pady=10)
                self.inputs[col] = spin

        # Predict Button at bottom outside the scrollable frame
        predict_btn = ctk.CTkButton(root, text="Predict", font=("Arial", 18), command=self.predict, width=200)
        predict_btn.pack(pady=20)

    def predict(self):
        user_input = {}
        for col, widget in self.inputs.items():
            value = widget.get()
            if X[col].dtype == 'object':
                user_input[col] = value
            else:
                try:
                    if col.lower() in ['year', 'doors']:
                        user_input[col] = int(value)
                    else:
                        user_input[col] = float(value)
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid value for {col}")
                    return

        input_df = pd.DataFrame([user_input])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        # Predict both
        price = reg_model.predict(input_encoded)[0]
        category = cls_model.predict(input_encoded)[0]

        messagebox.showinfo("Prediction", f"Predicted Price: ${price:.2f}\nCategory: {category}")

if __name__ == "__main__":
    root = ctk.CTk()
    app = VehiclePriceApp(root)
    root.mainloop()
