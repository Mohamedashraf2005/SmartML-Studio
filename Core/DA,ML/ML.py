import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ctk.set_appearance_mode("white")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.geometry("1200x700")
app.title("Data Analysis Models Interface")


selected_model = None
X, y = None, None
data = None
target_column = None
feature_columns = []


def select_model(name):
    global selected_model
    if name == "Decision Tree":
        selected_model = DecisionTreeClassifier()
    elif name == "KNN":
        selected_model = KNeighborsClassifier()
    elif name == "SVM":
        selected_model = SVC()
    elif name == "Naive Bayes":
        selected_model = GaussianNB()
    elif name == "Random Forest":
        selected_model = RandomForestClassifier()
    elif name == "KMeans":
        selected_model = KMeans(n_clusters=3)
    main_display.delete("0.0", "end")
    main_display.insert("0.0", f"Selected Model: {name}")

def load_data():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        update_feature_list()
        update_target_list()
        main_display.delete("0.0", "end")
        main_display.insert("0.0", f"Data Loaded: {file_path.split('/')[-1]}")

def update_feature_list():
    if data is not None:
        feature_listbox.delete(0, "end")
        for col in data.columns:
            feature_listbox.insert("end", col)

def update_target_list():
    if data is not None:
        target_menu.configure(values=list(data.columns))
        target_menu.set(data.columns[0])

def set_features():
    global feature_columns
    selected = feature_listbox.curselection()
    feature_columns = [feature_listbox.get(i) for i in selected]
    main_display.insert("end", f"\nSelected Features: {feature_columns}")

def set_target():
    global target_column
    target_column = target_menu.get()
    main_display.insert("end", f"\nSelected Target: {target_column}")

def prepare_data():
    global X, y
    if data is not None and feature_columns and target_column:
        X = data[feature_columns]
        y = data[target_column]
        main_display.insert("end", "\n Data is ready for training!")
    else:
        main_display.insert("end", "\n Please select features and target first!")

def evaluate_model():
    if selected_model is None:
        main_display.insert("end", "\n Please select a model first!")
        return
    
    prepare_data()
    if X is None or y is None:
        main_display.insert("end", "\n Data is not ready!")
        return
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)

    # Train/Test Metrics
    acc_split = accuracy_score(y_test, y_pred)
    prec_split = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_split = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_split = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_kf = cross_val_score(selected_model, X, y, cv=kf, scoring='accuracy')
    prec_kf = cross_val_score(selected_model, X, y, cv=kf, scoring='precision_weighted')
    rec_kf = cross_val_score(selected_model, X, y, cv=kf, scoring='recall_weighted')
    f1_kf = cross_val_score(selected_model, X, y, cv=kf, scoring='f1_weighted')

    result = (
        f"{'Metric':<15}{'Train/Test':<15}{'K-Fold (5)':<15}\n"
        f"{'-'*45}\n"
        f"{'Accuracy':<15}{acc_split:.4f}{np.mean(acc_kf):>15.4f}\n"
        f"{'Precision':<15}{prec_split:.4f}{np.mean(prec_kf):>15.4f}\n"
        f"{'Recall':<15}{rec_split:.4f}{np.mean(rec_kf):>15.4f}\n"
        f"{'F1-Score':<15}{f1_split:.4f}{np.mean(f1_kf):>15.4f}\n"
    )

    main_display.delete("0.0", "end")
    main_display.insert("0.0", result)

top_frame = ctk.CTkFrame(app, corner_radius=10)
top_frame.pack(side="top", fill="x", padx=100, pady=15)

models = ["Decision Tree", "KNN", "SVM", "Naive Bayes", "Random Forest", "KMeans"]
for model in models:
    ctk.CTkButton(top_frame, text=model, width=120, height=35, command=lambda m=model: select_model(m)).pack(side="left", padx=10, pady=10)

side_frame = ctk.CTkFrame(app, width=250, corner_radius=10)
side_frame.pack(side="left", fill="y", padx=20, pady=10)

ctk.CTkLabel(side_frame, text="Load & Prepare Data", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)
ctk.CTkButton(side_frame, text="Load CSV", width=200, command=load_data).pack(pady=5)

ctk.CTkLabel(side_frame, text="Select Features (multi-select)").pack(pady=5)
feature_listbox = tk.Listbox(side_frame, selectmode="multiple", height=10, width=30)
feature_listbox.pack(pady=5)

ctk.CTkButton(side_frame, text="Set Features", width=200, command=set_features).pack(pady=5)

ctk.CTkLabel(side_frame, text="Select Target Column").pack(pady=5)
target_menu = ctk.CTkComboBox(side_frame, width=200)
target_menu.pack(pady=5)

ctk.CTkButton(side_frame, text="Set Target", width=200, command=set_target).pack(pady=5)

ctk.CTkLabel(side_frame, text="Model Evaluation", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)
ctk.CTkButton(side_frame, text="Evaluate Model", width=200, command=evaluate_model).pack(pady=5)


main_display = ctk.CTkTextbox(app, width=700, height=400, corner_radius=10)
main_display.pack(padx=20, pady=20, expand=True)
main_display.insert("0.0", " Model results will appear here...")


app.mainloop()
