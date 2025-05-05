import customtkinter as ctk
import pandas as pd
from sklearn.model_selection import train_test_split

class ModelSelectionApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.dataset = None
        self.load_dataset()
        self.root.configure(fg_color="#435242")
        self.task = ctk.StringVar(value="Regression")
        self.target_var = ctk.StringVar()
        self.test_size_var = ctk.DoubleVar(value=0.2)
        self.random_state_var = ctk.IntVar(value=42)
        self.algorithm_var = ctk.StringVar(value="Logistic Regression")

        self.setup_layout()

    def load_dataset(self):
        try:
            self.dataset = self.main_app.shared_data.get("preprocessed_dataset")
            if self.dataset is not None and len(self.dataset.columns) > 0:
                self.target_var.set(self.dataset.columns[0])
            else:
                self.target_var.set("No dataset loaded")
            print("Loaded preprocessed dataset")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def setup_layout(self):
        self.container = ctk.CTkFrame(self.root, fg_color="#435242")
        self.container.pack(expand=True, fill="both", padx=10, pady=10)

        ctk.CTkLabel(
            self.container, text="Model Selection", font=("Arial", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 5))
        ctk.CTkLabel(
            self.container, text="Configure and train your machine learning model",
            font=("Arial", 14), text_color="#cccccc"
        ).pack(pady=(0, 20))

        main_frame = ctk.CTkFrame(self.container, fg_color="#435242")
        main_frame.pack(expand=True, fill="both", padx=20, pady=10)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)

        self.create_sidebar(main_frame)
        self.create_content_area(main_frame)

    def create_sidebar(self, parent):
        self.sidebar = ctk.CTkFrame(parent, fg_color="#435242")
        self.sidebar.place(relwidth=0.25, relheight=1.0)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 20))

        ctk.CTkLabel(
            self.sidebar, text="Model Configuration", font=("Arial", 16, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 10))

        # Task selection
        ctk.CTkLabel(self.sidebar, text="Choose Task:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        ctk.CTkOptionMenu(
            self.sidebar, values=["Regression", "Classification"], variable=self.task,
            fg_color="#445344", text_color="#ffffff", button_color="#445344", button_hover_color="#6b9463"
        ).pack(pady=5, padx=10, fill="x")

        # Target column
        ctk.CTkLabel(self.sidebar, text="Target Column:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        target_menu = ctk.CTkOptionMenu(
            self.sidebar, values=list(self.dataset.columns) if self.dataset is not None else ["No dataset loaded"],
            variable=self.target_var, fg_color="#445344", text_color="#ffffff",
            button_color="#445344", button_hover_color="#6b9463"
        )
        target_menu.pack(pady=5, padx=10, fill="x")

        # Train-test split
        ctk.CTkLabel(self.sidebar, text="Train Test Size:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        ctk.CTkSlider(
            self.sidebar, variable=self.test_size_var, from_=0.1, to=0.5, number_of_steps=4,
            fg_color="#445344", progress_color="#5e7a51"
        ).pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(self.sidebar, text="Random State:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        ctk.CTkEntry(
            self.sidebar, textvariable=self.random_state_var, fg_color="#445344", text_color="#ffffff",
            border_color="#5e7a51", corner_radius=10
        ).pack(pady=5, padx=10, fill="x")

        # Algorithm selection
        ctk.CTkLabel(self.sidebar, text="Choose Algorithm:", font=("Arial", 14), text_color="#ffffff").pack(pady=(10, 5))
        algorithms = ["Logistic Regression", "Random Forest", "SVM", "Decision Tree"]
        ctk.CTkOptionMenu(
            self.sidebar, values=algorithms, variable=self.algorithm_var,
            fg_color="#445344", text_color="#ffffff", button_color="#445344", button_hover_color="#6b9463"
        ).pack(pady=5, padx=10, fill="x")

        train_button = ctk.CTkButton(
            self.sidebar, text="Train Model", width=180, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=self.train_model
        )
        train_button.pack(pady=20, padx=10, fill="x")

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

        nav_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        nav_frame.pack(pady=10)

        back_button = ctk.CTkButton(
            nav_frame, text="Back to Preprocessing", width=200, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=lambda: self.main_app.show_page("Preprocessing")
        )
        back_button.pack()

    def train_model(self):
        if self.dataset is None:
            ctk.CTkLabel(
                self.content_area, text="No dataset loaded. Please preprocess a dataset first.",
                font=("Arial", 16), text_color="#cccccc", wraplength=300
            ).pack(pady=(20, 10))
            return

        for widget in self.content_area.winfo_children():
            widget.destroy()

        target_column = self.target_var.get()
        if target_column not in self.dataset.columns or target_column == "No dataset loaded":
            ctk.CTkLabel(
                self.content_area, text="Invalid target column selected.",
                font=("Arial", 16), text_color="#cccccc", wraplength=300
            ).pack(pady=(20, 10))
            return

        X = self.dataset.drop(columns=[target_column])
        y = self.dataset[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size_var.get(), random_state=self.random_state_var.get()
        )

        self.main_app.shared_data["trained_model"] = {
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "algorithm": self.algorithm_var.get(), "task": self.task.get()
        }
        ctk.CTkLabel(
            self.content_area, text=f"Model trained with {self.algorithm_var.get()} on {self.task.get()} task.",
            font=("Arial", 16), text_color="#ffffff", wraplength=300
        ).pack(pady=(20, 10))
        self.main_app.show_page("Dashboard")
