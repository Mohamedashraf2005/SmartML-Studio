import customtkinter as ctk
import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            accuracy_score, precision_score, recall_score, f1_score,
                            silhouette_score, davies_bouldin_score, calinski_harabasz_score)

class ModelEvaluationApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.root.configure(fg_color="#435242")
        self.trained_model = None
        self.selected_features = []
        self.dataset = None

        # Define status_label before packing
        self.status_label = ctk.CTkLabel(
            self.root, text="Initializing...", font=("Arial", 14), text_color="#cccccc"
        )
        self.status_label.pack(pady=10)

        self.setup_layout()
        self.load_data()

    def load_data(self):
        """Load the trained model, selected features, and preprocessed dataset."""
        try:
            self.trained_model = self.main_app.shared_data.get("trained_model")
            if self.trained_model:
                self.selected_features = self.trained_model.get("features", [])
                print(f"Loaded trained model: {self.trained_model['algorithm']}")
                print(f"Selected features: {self.selected_features}")
            else:
                print("No trained model found in shared data.")
                self.status_label.configure(text="No trained model available", text_color="#ff0000")

            self.dataset = self.main_app.shared_data.get("preprocessed_dataset")
            if self.dataset is not None:
                print(f"Preprocessed dataset loaded: ({self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns)")
            else:
                print("No preprocessed dataset found in shared data.")
                self.status_label.configure(text="No dataset available", text_color="#ff0000")

            self.update_feature_list()
            self.update_ui_state()
            self.create_prediction_inputs()

        except Exception as e:
            print(f"Error loading data: {e}")
            self.status_label.configure(text=f"Error loading data: {e}", text_color="#ff0000")

    def update_ui_state(self):
        """Update UI based on loaded data."""
        if self.trained_model and self.dataset is not None:
            self.status_label.configure(text="Model and dataset loaded successfully", text_color="#00ff00")
        else:
            self.status_label.configure(
                text="Model or dataset missing. Please train a model and preprocess a dataset.",
                text_color="#ff0000"
            )

    def setup_layout(self):
        """Set up the main layout with a sidebar and content area."""
        # Main scrollable container
        self.container = ctk.CTkScrollableFrame(
            self.root, fg_color="#435242", corner_radius=10,
            scrollbar_button_color="#4a4a4a", scrollbar_button_hover_color="#666666"
        )
        self.container.pack(expand=True, fill="both", padx=10, pady=10)

        # Title and subtitle
        ctk.CTkLabel(
            self.container, text="Model Evaluation & Prediction",
            font=("Arial", 24, "bold"), text_color="#ffffff"
        ).pack(pady=(10, 5))
        ctk.CTkLabel(
            self.container, text="Evaluate your model and make predictions",
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
            nav_frame, text="Back to Model Selection", width=200, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=lambda: self.main_app.show_page("ModelSelection")
        )
        back_button.pack()

    def create_sidebar(self, parent):
        """Create the sidebar to display selected features and a button to show output."""
        self.sidebar = ctk.CTkFrame(parent, fg_color="#435242", width=250)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 20))

        ctk.CTkLabel(
            self.sidebar, text="Selected Features", font=("Arial", 16, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 10))

        # Frame for the feature list
        self.feature_frame = ctk.CTkFrame(self.sidebar, fg_color="#445344", corner_radius=10)
        self.feature_frame.pack(fill="x", padx=10, pady=5)

        self.feature_labels = []
        # Placeholder label if no features are loaded
        self.no_features_label = ctk.CTkLabel(
            self.feature_frame, text="No features selected.",
            font=("Arial", 14), text_color="#cccccc"
        )
        self.no_features_label.pack(pady=10)

        # Show Output button
        show_output_button = ctk.CTkButton(
            self.sidebar, text="Show Output", width=180, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=self.show_evaluation
        )
        show_output_button.pack(pady=20, padx=10, fill="x")

    def update_feature_list(self):
        """Update the sidebar with the list of selected features."""
        # Clear existing labels
        for label in self.feature_labels:
            label.destroy()
        self.feature_labels.clear()
        self.no_features_label.pack_forget()

        if not self.selected_features:
            self.no_features_label.pack(pady=10)
        else:
            for feature in self.selected_features:
                label = ctk.CTkLabel(
                    self.feature_frame, text=feature, font=("Arial", 14),
                    text_color="#ffffff", wraplength=200
                )
                label.pack(pady=5, padx=10, anchor="w")
                self.feature_labels.append(label)

    def create_content_area(self, parent):
        """Create the content area for evaluation and prediction sections."""
        self.content_area = ctk.CTkFrame(parent, fg_color="#445344", corner_radius=15, border_width=3, border_color="#3a473a")
        self.content_area.grid(row=0, column=1, sticky="nsew")
        self.content_area.grid_rowconfigure(0, weight=1)
        self.content_area.grid_columnconfigure(0, weight=1)

        self.content_area.bind("<Enter>", lambda e: self.content_area.configure(fg_color="#5e7a51"))
        self.content_area.bind("<Leave>", lambda e: self.content_area.configure(fg_color="#445344"))

        # Scrollable frame for evaluation
        self.eval_frame = ctk.CTkScrollableFrame(
            self.content_area, fg_color="transparent", corner_radius=10,
            scrollbar_button_color="#4a4a4a", scrollbar_button_hover_color="#666666"
        )
        self.eval_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            self.eval_frame, text="Model Evaluation", font=("Arial", 16, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 10))

        self.eval_output_label = ctk.CTkLabel(
            self.eval_frame, text="Click 'Show Output' to evaluate the model.",
            font=("Arial", 14), text_color="#cccccc", wraplength=350
        )
        self.eval_output_label.pack(pady=10)

        # Prediction section
        self.prediction_frame = ctk.CTkFrame(self.content_area, fg_color="#445344", corner_radius=10)
        self.prediction_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            self.prediction_frame, text="Make a Prediction", font=("Arial", 16, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 10))

        self.input_frame = ctk.CTkFrame(self.prediction_frame, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10)

        self.input_entries = {}
        self.create_prediction_inputs()

        predict_button = ctk.CTkButton(
            self.prediction_frame, text="Predict", width=180, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=self.make_prediction
        )
        predict_button.pack(pady=10)

        self.prediction_result_label = ctk.CTkLabel(
            self.prediction_frame, text="Prediction result will appear here.",
            font=("Arial", 14), text_color="#cccccc", wraplength=350
        )
        self.prediction_result_label.pack(pady=10)

    def create_prediction_inputs(self):
        """Create input fields for each selected feature."""
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.input_entries.clear()

        if not self.selected_features or self.dataset is None:
            ctk.CTkLabel(
                self.input_frame, text="No features or dataset available for prediction.",
                font=("Arial", 14), text_color="#cccccc"
            ).pack(pady=10)
            return

        for feature in self.selected_features:
            frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
            frame.pack(fill="x", pady=5)

            ctk.CTkLabel(
                frame, text=f"{feature}:", font=("Arial", 14),
                text_color="#ffffff", width=150, anchor="w"
            ).pack(side="left", padx=(0, 10))

            # Check if the feature is categorical
            if self.dataset[feature].dtype == 'object' or pd.api.types.is_categorical_dtype(self.dataset[feature]):
                unique_values = self.dataset[feature].unique().tolist()
                entry = ctk.CTkOptionMenu(
                    frame, values=unique_values, fg_color="#445344", text_color="#ffffff",
                    button_color="#445344", button_hover_color="#6b9463"
                )
            else:
                entry = ctk.CTkEntry(
                    frame, fg_color="#445344", text_color="#ffffff",
                    border_color="#5e7a51", corner_radius=10
                )
            entry.pack(side="left", fill="x", expand=True)
            self.input_entries[feature] = entry

    def show_evaluation(self):
        """Calculate and display evaluation metrics based on the model type."""
        for widget in self.eval_frame.winfo_children():
            widget.destroy()

        ctk.CTkLabel(
            self.eval_frame, text="Model Evaluation", font=("Arial", 16, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 10))

        if not self.trained_model:
            ctk.CTkLabel(
                self.eval_frame, text="No trained model available. Please train a model first.",
                font=("Arial", 14), text_color="#ff0000", wraplength=350
            ).pack(pady=10)
            return

        learning_type = self.trained_model.get("learning_type")
        summary_text = ""

        try:
            if learning_type == "supervised":
                task = self.trained_model.get("task")
                model = self.trained_model.get("model")
                X_test = self.trained_model.get("X_test")
                y_test = self.trained_model.get("y_test")

                y_pred = model.predict(X_test)

                if task == "Regression":
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    summary_text = (
                        f"        === Model Evaluation ===\n\n"
                        f"    --- Model Details ---\n"
                        f"    Model Type:\tSupervised\n"
                        f"    Task:\t\tRegression\n"
                        f"    Algorithm:\t{self.trained_model['algorithm']}\n"
                        f"    --- Evaluation Metrics ---\n"
                        f"    MAE:\t\t{mae:.4f}\n"
                        f"    MSE:\t\t{mse:.4f}\n"
                        f"    RMSE:\t\t{rmse:.4f}\n"
                        f"    R² Score:\t\t{r2:.4f}\n"
                    )

                elif task == "Classification":
                    accuracy = accuracy_score(y_test,y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    summary_text = (
                        f"        === Model Evaluation ===\n\n"
                        f"    --- Model Details ---\n"
                        f"    Model Type:\tSupervised\n"
                        f"    Task:\t\tClassification\n"
                        f"    Algorithm:\t{self.trained_model['algorithm']}\n"
                        f"    --- Evaluation Metrics ---\n"
                        f"    Accuracy:\t\t{accuracy:.4f}\n"
                        f"    Precision:\t\t{precision:.4f}\n"
                        f"    Recall:\t\t{recall:.4f}\n"
                        f"    F1 Score:\t\t{f1:.4f}\n"
                    )

            else:  # Unsupervised (Clustering)
                model = self.trained_model.get("model")
                X = self.dataset[self.selected_features]
                labels = self.trained_model.get("labels")

                silhouette = silhouette_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)
                calinski_harabasz = calinski_harabasz_score(X, labels)

                summary_text = (
                    f"        === Model Evaluation ===\n\n"
                    f"    --- Model Details ---\n"
                    f"    Model Type:\tUnsupervised\n"
                    f"    Algorithm:\t{self.trained_model['algorithm']}\n"
                    f"    --- Evaluation Metrics ---\n"
                    f"    Silhouette Score:\t\t{silhouette:.4f}\n"
                    f"    Davies-Bouldin Index:\t\t{davies_bouldin:.4f}\n"
                    f"    Calinski-Harabasz Index:\t\t{calinski_harabasz:.4f}\n"
                )

            ctk.CTkLabel(
                self.eval_frame, text=summary_text, font=("Arial", 14, "bold"),
                text_color="#ffffff", wraplength=350, justify="center"
            ).pack(pady=10, padx=10, fill="both", expand=True)

        except Exception as e:
            ctk.CTkLabel(
                self.eval_frame, text=f"Error evaluating model: {e}",
                font=("Arial", 14), text_color="#ff0000", wraplength=350
            ).pack(pady=10)

    def make_prediction(self):
        """Collect input values, make prediction using trained model, and display result."""
        if not self.trained_model or "model" not in self.trained_model:
            self.eval_output_label.configure(text="No trained model available.", text_color="#ff0000")
            return

        model = self.trained_model["model"]
        input_values = []

        try:
            for feature in self.selected_features:
                value = self.input_entries[feature].get()
                if value == "":
                    raise ValueError(f"Value for '{feature}' is missing.")
                input_values.append(float(value))  # Convert to float (or int if needed)

            # Create DataFrame with one row and the selected features as columns
            input_df = pd.DataFrame([input_values], columns=self.selected_features)

            # Predict using model
            prediction = model.predict(input_df)[0]

            self.eval_output_label.configure(
                text=f"Prediction result: {prediction}", text_color="#00ff00"
            )
        except Exception as e:
            self.eval_output_label.configure(
                text=f"Error in prediction: {e}", text_color="#ff0000"
            )
