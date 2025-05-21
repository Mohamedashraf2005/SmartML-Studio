import customtkinter as ctk
import pandas as pd
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, r2_score
import time

class ModelSelectionApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.dataset = None
        self.load_dataset()
        self.root.configure(fg_color="#435242")
        self.learning_type = ctk.StringVar(value="supervised")
        self.task = ctk.StringVar(value="Regression")
        self.target_var = ctk.StringVar()
        self.test_size_var = ctk.DoubleVar(value=0.2)
        self.random_state_var = ctk.IntVar(value=42)
        self.algorithm_var = ctk.StringVar(value="Linear Regression")
        self.task_menu = None
        self.algorithm_menu = None
        self.feature_listbox = None  # For multi-select features
        self.selected_features = []  # Store selected feature names

        self.setup_layout()

    def load_dataset(self):
        try:
            self.dataset = self.main_app.shared_data.get("preprocessed_dataset")
            if self.dataset is not None and len(self.dataset.columns) > 0:
                self.target_var.set(self.dataset.columns[0])
            else:
                self.target_var.set("No dataset loaded")
            print("Loaded preprocessed dataset")
            self.update_ui_state()  # Update UI after loading dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def update_ui_state(self):
        """Update the UI state based on the current dataset and selections."""
        if self.dataset is not None and not self.dataset.empty:
            self.target_menu.configure(values=list(self.dataset.columns), state="normal")
            if self.feature_listbox:
                self.feature_listbox.delete(0, tk.END)
                for col in self.dataset.columns:
                    self.feature_listbox.insert(tk.END, col)
            if self.dataset.columns.size > 0 and (not self.target_var.get() or self.target_var.get() not in self.dataset.columns):
                self.target_var.set(self.dataset.columns[0])
        else:
            self.target_menu.configure(values=["No dataset loaded"], state="disabled")
            if self.feature_listbox:
                self.feature_listbox.delete(0, tk.END)
                self.feature_listbox.insert(tk.END, "No dataset loaded")
            self.target_var.set("No dataset loaded")
        self.update_task_and_algorithm_menu(self.learning_type.get())

    def setup_layout(self):
        # Create a scrollable frame as the main container (only scrollbar on the far right)
        self.container = ctk.CTkScrollableFrame(
            self.root, fg_color="#435242", corner_radius=10,
            scrollbar_button_color="#4a4a4a", scrollbar_button_hover_color="#666666"
        )
        self.container.pack(expand=True, fill="both", padx=10, pady=10)

        # Title and subtitle
        ctk.CTkLabel(
            self.container, text="Model Selection", font=("Arial", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 5))
        ctk.CTkLabel(
            self.container, text="Configure and train your machine learning model",
            font=("Arial", 14), text_color="#cccccc"
        ).pack(pady=(0, 20))

        # Main frame for sidebar and content area
        main_frame = ctk.CTkFrame(self.container, fg_color="#435242")
        main_frame.pack(expand=True, fill="both", padx=20, pady=10)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)

        self.create_sidebar(main_frame)
        self.create_content_area(main_frame)

        # Navigation frame
        nav_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        nav_frame.pack(pady=10, fill="x")

        back_button = ctk.CTkButton(
            nav_frame, text="Back to Preprocessing", width=200, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=lambda: self.main_app.show_page("Preprocessing")
        )
        back_button.pack()

    def create_sidebar(self, parent):
        # Use a regular frame for the sidebar (no scrollbar here)
        self.sidebar = ctk.CTkFrame(parent, fg_color="#435242", width=250)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 20))

        ctk.CTkLabel(
            self.sidebar, text="Model Configuration", font=("Arial", 16, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 10))

        # Learning type selection
        ctk.CTkLabel(self.sidebar, text="Choose Type:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        ctk.CTkOptionMenu(
            self.sidebar, values=["supervised", "unsupervised"], variable=self.learning_type,
            fg_color="#445344", text_color="#ffffff", button_color="#445344", button_hover_color="#6b9463",
            command=self.update_task_and_algorithm_menu
        ).pack(pady=5, padx=10, fill="x")

        # Task selection (only for supervised)
        self.task_label = ctk.CTkLabel(self.sidebar, text="Choose Task:", font=("Arial", 14), text_color="#ffffff")
        self.task_label.pack(pady=(10, 5))
        self.task_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["Regression", "Classification"], variable=self.task,
            fg_color="#445344", text_color="#ffffff", button_color="#445344", button_hover_color="#6b9463",
            command=self.update_algorithm_menu
        )
        self.task_menu.pack(pady=5, padx=10, fill="x")

        # Target column
        ctk.CTkLabel(self.sidebar, text="Target Column:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        self.target_menu = ctk.CTkOptionMenu(
            self.sidebar, values=list(self.dataset.columns) if self.dataset is not None else ["No dataset loaded"],
            variable=self.target_var, fg_color="#445344", text_color="#ffffff",
            button_color="#445344", button_hover_color="#6b9463"
        )
        self.target_menu.pack(pady=5, padx=10, fill="x")

        # Feature selection (multi-select)
        ctk.CTkLabel(self.sidebar, text="Select Features:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        self.feature_listbox = tk.Listbox(
            self.sidebar, selectmode="multiple", height=6, width=30,
            bg="#445344", fg="#ffffff", selectbackground="#5e7a51", selectforeground="#ffffff",
            font=("Arial", 12)
        )
        self.feature_listbox.pack(pady=5, padx=10, fill="x")
        if self.dataset is not None:
            for col in self.dataset.columns:
                self.feature_listbox.insert(tk.END, col)

        # Train-test split
        ctk.CTkLabel(self.sidebar, text="Train Test Size:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        self.test_size_slider = ctk.CTkSlider(
            self.sidebar, variable=self.test_size_var, from_=0.1, to=0.5, number_of_steps=4,
            fg_color="#445344", progress_color="#5e7a51"
        )
        self.test_size_slider.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(self.sidebar, text="Random State:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        self.random_state_entry = ctk.CTkEntry(
            self.sidebar, textvariable=self.random_state_var, fg_color="#445344", text_color="#ffffff",
            border_color="#5e7a51", corner_radius=10
        )
        self.random_state_entry.pack(pady=5, padx=10, fill="x")

        # Algorithm selection
        ctk.CTkLabel(self.sidebar, text="Choose Algorithm:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        algorithms = ["Linear Regression", "Multiple Regression"]
        self.algorithm_menu = ctk.CTkOptionMenu(
            self.sidebar, values=algorithms, variable=self.algorithm_var,
            fg_color="#445344", text_color="#ffffff", button_color="#445344", button_hover_color="#6b9463"
        )
        self.algorithm_menu.pack(pady=5, padx=10, fill="x")

        train_button = ctk.CTkButton(
            self.sidebar, text="Train Model", width=180, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=self.train_model
        )
        train_button.pack(pady=20, padx=10, fill="x")

        # Update UI based on initial learning type
        self.update_task_and_algorithm_menu(self.learning_type.get())

    def update_task_and_algorithm_menu(self, learning_type):
        """Update task and algorithm menus based on learning type"""
        if learning_type == "supervised":
            self.task_label.pack(pady=(10, 5))
            self.task_menu.pack(pady=5, padx=10, fill="x")
            self.target_menu.configure(state="normal")
            self.test_size_slider.configure(state="normal")
            self.random_state_entry.configure(state="normal")
            self.feature_listbox.configure(state="normal")
        else:  # unsupervised
            self.task_label.pack_forget()
            self.task_menu.pack_forget()
            self.target_menu.configure(state="disabled")
            self.test_size_slider.configure(state="disabled")
            self.random_state_entry.configure(state="disabled")
            self.feature_listbox.configure(state="normal")  # Still allow feature selection
            self.algorithm_menu.configure(values=["Clustering"])
            self.algorithm_var.set("Clustering")

    def update_algorithm_menu(self, task):
        """Update algorithm menu based on task"""
        if self.learning_type.get() == "supervised":
            if task == "Regression":
                algorithms = ["Linear Regression", "Multiple Regression"]
            else:  # Classification
                algorithms = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Naive Bayes"]
            self.algorithm_menu.configure(values=algorithms)
            self.algorithm_var.set(algorithms[0])

    def create_content_area(self, parent):
        self.content_area = ctk.CTkFrame(parent, fg_color="#445344", corner_radius=15, border_width=3, border_color="#3a473a")
        self.content_area.grid(row=0, column=1, sticky="nsew")
        self.content_area.grid_rowconfigure(0, weight=1)
        self.content_area.grid_columnconfigure(0, weight=1)

        self.content_area.bind("<Enter>", lambda e: self.content_area.configure(fg_color="#5e7a51"))
        self.content_area.bind("<Leave>", lambda e: self.content_area.configure(fg_color="#445344"))

        ctk.CTkLabel(
            self.content_area, text="Configure your model and click Train to proceed",
            font=("Arial", 16), text_color="#ffffff", wraplength=300
        ).pack(pady=(20, 10))

    def train_model(self):
        if self.dataset is None or self.dataset.empty:
            for widget in self.content_area.winfo_children():
                widget.destroy()
            ctk.CTkLabel(
                self.content_area, text="No dataset loaded. Please preprocess a dataset first.",
                font=("Arial", 16), text_color="#cccccc", wraplength=300
            ).pack(pady=(20, 10))
            return

        # Get selected features and store them in a list
        selected_indices = self.feature_listbox.curselection()
        self.selected_features = [self.feature_listbox.get(i) for i in selected_indices]
        if not self.selected_features:
            for widget in self.content_area.winfo_children():
                widget.destroy()
            ctk.CTkLabel(
                self.content_area, text="Please select at least one feature for training.",
                font=("Arial", 16), text_color="#ff0000", wraplength=300
            ).pack(pady=(20, 10))
            return
        else:
            print("Selected features:", self.selected_features)  # Print selected features to terminal

        for widget in self.content_area.winfo_children():
            widget.destroy()

        summary_text = ""
        if self.learning_type.get() == "supervised":
            target_column = self.target_var.get()
            if target_column not in self.dataset.columns or target_column == "No dataset loaded":
                ctk.CTkLabel(
                    self.content_area, text="Invalid target column selected.",
                    font=("Arial", 16), text_color="#cccccc", wraplength=300
                ).pack(pady=(20, 10))
                return

            # Ensure target column is not in selected features
            if target_column in self.selected_features:
                self.selected_features.remove(target_column)

            if not self.selected_features:
                ctk.CTkLabel(
                    self.content_area, text="Please select at least one feature excluding the target column.",
                    font=("Arial", 16), text_color="#ff0000", wraplength=300
                ).pack(pady=(20, 10))
                return

            X = self.dataset[self.selected_features]
            y = self.dataset[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size_var.get(), random_state=self.random_state_var.get()
            )

            algorithm = self.algorithm_var.get()
            if algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Multiple Regression":
                model = LinearRegression()
            elif algorithm == "Logistic Regression":
                model = LogisticRegression()
            elif algorithm == "KNN":
                model = KNeighborsClassifier()
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier()
            elif algorithm == "Random Forest":
                model = RandomForestClassifier()
            elif algorithm == "SVM":
                model = SVC()
            elif algorithm == "Naive Bayes":
                model = GaussianNB()
            else:
                ctk.CTkLabel(
                    self.content_area, text="Invalid algorithm selected.",
                    font=("Arial", 16), text_color="#cccccc", wraplength=300
                ).pack(pady=(20, 10))
                return

            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                self.main_app.shared_data["trained_model"] = {
                    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                    "algorithm": algorithm, "task": self.task.get(), "learning_type": self.learning_type.get(),
                    "model": model, "features": self.selected_features
                }

                summary_text = (
                    f"        === Model Trained Successfully ===\n\n"
                    f"    --- Model Details ---\n"
                    f"    Model Type:\tSupervised\n"
                    f"    Task:\t\t{self.task.get()}\n"
                    f"    Algorithm:\t{algorithm}\n"
                    f"    --- Performance Metrics ---\n"
                    f"    Train-Test Split:\t{1 - self.test_size_var.get():.2f}/{self.test_size_var.get():.2f}\n"
                    f"    Testing Accuracy:\t{test_score:.4f}\n"
                )

                # Feature importance for tree-based models (commented out as per your snippet)
                if algorithm in ["Decision Tree", "Random Forest"]:
                    feature_importance = model.feature_importances_
                    importance_dict = dict(zip(self.selected_features, feature_importance))
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]  # Top 3 features
                    # summary_text += f"\n    --- Top 3 Feature Importance ---\n"
                    # for feature, importance in sorted_importance:
                    #     summary_text += f"    Feature: {feature}\tImportance: {importance:.4f}\n"

                if self.task.get() == "Classification":
                    y_pred = model.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=False)
                    summary_text += f"\n        --- Classification Report ---\n{report}"

                # Display summary directly in content area (no nested scrollbar)
                ctk.CTkLabel(
                    self.content_area, text=summary_text, font=("Arial", 15, "bold"), text_color="#ffffff",
                    wraplength=350, justify="center", anchor="center"
                ).pack(pady=10, padx=10, fill="both", expand=True)

            except Exception as e:
                ctk.CTkLabel(
                    self.content_area, text=f"Error training model: {e}",
                    font=("Arial", 16), text_color="#ff0000", wraplength=300
                ).pack(pady=(20, 10))
                return

        else:  # unsupervised
            algorithm = self.algorithm_var.get()
            if algorithm == "Clustering":
                X = self.dataset[self.selected_features]
                start_time = time.time()
                model = KMeans(n_clusters=3, random_state=self.random_state_var.get())
                labels = model.fit_predict(X)
                training_time = time.time() - start_time
                self.main_app.shared_data["trained_model"] = {
                    "labels": labels, "algorithm": algorithm, "learning_type": self.learning_type.get(),
                    "model": model, "features": self.selected_features
                }
                inertia = model.inertia_
                summary_text = (
                    f"\t\t=== Model Trained Successfully ===\n\n"
                    f"--- Model Details ---\n"
                    f"Model Type:\tUnsupervised\n"
                    f"Algorithm:\t{algorithm}\n"
                    f"Training Time:\t{training_time:.2f} seconds\n\n"
                    f"--- Performance Metrics ---\n"
                    f"Inertia:\t{inertia:.4f}\n"
                )

                # Display summary directly in content area (no nested scrollbar)
                ctk.CTkLabel(
                    self.content_area, text=summary_text, font=("Arial", 12, "bold"), text_color="#ffffff",
                    wraplength=350, justify="left", anchor="nw"
                ).pack(pady=10, padx=10, fill="both", expand=True)
            else:
                ctk.CTkLabel(
                    self.content_area, text="Invalid algorithm selected.",
                    font=("Arial", 16), text_color="#cccccc", wraplength=300
                ).pack(pady=(20, 10))
                return