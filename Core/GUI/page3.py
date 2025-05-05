import customtkinter as ctk
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

class DataPreprocessingApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.dataset = None
        self.load_dataset()
        self.selections = {}
        self.root.configure(fg_color="#435242")
        self.create_widgets()

    def load_dataset(self):
        try:
            file_path = self.main_app.shared_data.get("dataset_path")
            if file_path:
                if file_path.endswith(".csv"):
                    self.dataset = pd.read_csv(file_path)
                elif file_path.endswith(".xlsx"):
                    self.dataset = pd.read_excel(file_path)
                print(f"Loaded dataset: {file_path}")
            else:
                print("No dataset found")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def create_widgets(self):
        self.container = ctk.CTkFrame(self.root, fg_color="#435242")
        self.container.pack(expand=True, fill="both", padx=10, pady=10)

        ctk.CTkLabel(
            self.container, text="Data Preprocessing Options", font=("Arial", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 5))
        ctk.CTkLabel(
            self.container, text="Select methods to preprocess your dataset",
            font=("Arial", 14), text_color="#cccccc"
        ).pack(pady=(0, 20))

        cards_container = ctk.CTkFrame(self.container, fg_color="#435242")
        cards_container.pack(expand=True, fill="both", padx=20, pady=10)

        cards_container.grid_rowconfigure((0, 1), weight=1)
        cards_container.grid_columnconfigure((0, 1), weight=1)

        cards_data = [
            ("Missing Values:", ["Drop", "Fill", "Simple Imputer"]),
            ("Categorical Encoding:", ["One-Hot", "Label Encoding"]),
            ("Normalization:", ["Min-Max", "Z-score"]),
            ("Outliers:", ["Remove", "Cap", "Transform", "Ignore"]),
        ]

        for idx, (label, options) in enumerate(cards_data):
            row, col = divmod(idx, 2)
            card = self.create_card(cards_container, label, options)
            card.grid(row=row, column=col, padx=20, pady=20, sticky="nsew")

        apply_button = ctk.CTkButton(
            self.container, text="Apply Preprocessing", width=200, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=self.apply_preprocessing
        )
        apply_button.pack(pady=10)

        nav_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        nav_frame.pack(pady=10)

        back_button = ctk.CTkButton(
            nav_frame, text="Back to Dashboard", width=200, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=lambda: self.main_app.show_page("Dashboard")
        )
        back_button.pack(side="left", padx=10)

        next_button = ctk.CTkButton(
            nav_frame, text="Next Page", width=200, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14), command=lambda: self.main_app.show_page("ModelSelection")
        )
        next_button.pack(side="left", padx=10)

    def create_card(self, parent, text, options):
        frame = ctk.CTkFrame(
            parent, fg_color="#445344", corner_radius=15,
            border_width=3, border_color="#3a473a"
        )
        ctk.CTkLabel(
            frame, text=text, font=("Arial", 16), text_color="#ffffff", wraplength=200
        ).pack(pady=(20, 10))
        option_menu = ctk.CTkOptionMenu(
            frame, values=options, fg_color="#445344", text_color="#ffffff",
            button_color="#445344", button_hover_color="#6b9463",
            command=lambda value: self.selections.update({text: value})
        )
        option_menu.pack(pady=(0, 20))
        frame.bind("<Enter>", lambda e: frame.configure(fg_color="#5e7a51"))
        frame.bind("<Leave>", lambda e: frame.configure(fg_color="#445344"))
        self.selections[text] = options[0]
        return frame

    def apply_preprocessing(self):
        if self.dataset is None:
            print("No dataset loaded")
            return
        df = self.dataset.copy()
        
        if self.selections.get("Missing Values:") == "Drop":
            df = df.dropna()
        elif self.selections.get("Missing Values:") == "Fill":
            df = df.fillna(df.mean(numeric_only=True))
        elif self.selections.get("Missing Values:") == "Simple Imputer":
            imputer = SimpleImputer(strategy="mean")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        if self.selections.get("Categorical Encoding:") == "One-Hot":
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(df[categorical_cols])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
                df = pd.concat([df.drop(categorical_cols, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        elif self.selections.get("Categorical Encoding:") == "Label Encoding":
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

        if self.selections.get("Normalization:") == "Min-Max":
            scaler = MinMaxScaler()
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif self.selections.get("Normalization:") == "Z-score":
            scaler = StandardScaler()
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        if self.selections.get("Outliers:") == "Remove":
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

        self.main_app.shared_data["preprocessed_dataset"] = df
        print("Preprocessing applied successfully")
