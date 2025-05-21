import customtkinter as ctk
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from pandastable import Table
import tkinter as tk

class DataPreprocessingApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.dataset = None
        self.preprocessed_dataset = None
        self.selections = {}
        self.processing_cards = []
        self.root.configure(fg_color="#435242")
        self.load_dataset()
        self.create_widgets()

    def load_dataset(self):
        try:
            file_path = self.main_app.shared_data.get("dataset_path")
            if not file_path:
                self.dataset = None
                return
            if file_path.endswith(".csv"):
                self.dataset = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                self.dataset = pd.read_excel(file_path)
            print(f"Dataset loaded successfully in Preprocessing: {file_path} ({self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns)")
            self.update_column_menus()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.dataset = None

    def create_widgets(self):
        self.container = ctk.CTkScrollableFrame(self.root, fg_color="#435242")
        self.container.pack(expand=True, fill="both", padx=10, pady=10)

        ctk.CTkLabel(
            self.container, text="Data Preprocessing Options", font=("Arial", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 5))
        ctk.CTkLabel(
            self.container, text="Select methods to preprocess your dataset",
            font=("Arial", 14), text_color="#cccccc"
        ).pack(pady=(0, 10))

        self.data_display_frame = ctk.CTkFrame(self.container, fg_color="#445344", corner_radius=15, border_width=3, border_color="#3a473a")
        self.data_display_frame.pack(fill="x", padx=20, pady=0)

        self.data_table = None

        button_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        button_frame.pack(pady=10)

        ctk.CTkButton(
            button_frame, text="Show Raw Dataset", width=180, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14, "bold"), command=self.show_raw_dataset
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame, text="Show Preprocessed Dataset", width=180, height=40,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14, "bold"), command=self.show_preprocessed_dataset
        ).pack(side="left", padx=10)

        cards_container = ctk.CTkFrame(self.container, fg_color="#435242")
        cards_container.pack(fill="x", padx=20, pady=10)

        cards_container.grid_rowconfigure((0, 1), weight=1)
        cards_container.grid_columnconfigure((0, 1), weight=1)

        cards_data = [
            ("Missing Values:", ["Categorical (Mode)", "Numerical (Drop)", "Numerical (Fill)", "Numerical (Simple Imputer)"]),
            ("Categorical Encoding:", ["One-Hot", "Label Encoding"]),
            ("Normalization:", ["Min-Max", "Z-score"]),
            ("Outliers:", ["Remove", "Cap", "Transform", "Ignore"]),
        ]

        for idx, (label, options) in enumerate(cards_data):
            row, col = divmod(idx, 2)
            card = self.create_card(cards_container, label, options)
            card.grid(row=row, column=col, padx=15, pady=8, sticky="nsew")
            self.processing_cards.append(card)

        apply_button = ctk.CTkButton(
            self.container, text="Apply All Preprocessing", width=250, height=50,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 16, "bold"), command=self.apply_preprocessing
        )
        apply_button.pack(pady=10)

        nav_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        nav_frame.pack(pady=10, fill="x")

        back_button = ctk.CTkButton(
            nav_frame, text="Back to Dashboard", width=180, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14, "bold"), command=lambda: self.main_app.show_page("Dashboard")
        )
        back_button.pack(side="left", padx=10)

        next_button = ctk.CTkButton(
            nav_frame, text="Next Page", width=180, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 14, "bold"), command=lambda: self.main_app.show_page("Model Evaluation")
        )
        next_button.pack(side="left", padx=10)

    def create_card(self, parent, text, options):
        frame = ctk.CTkFrame(
            parent, fg_color="#445344", corner_radius=15,
            border_width=3, border_color="#3a473a"
        )

        ctk.CTkLabel(
            frame, text=text, font=("Arial", 16, "bold"), text_color="#ffffff", wraplength=150
        ).pack(pady=(15, 5))

        self.selections[text] = {"method": options[0], "column": None}

        method_menu = ctk.CTkOptionMenu(
            frame, values=options, fg_color="#445344", text_color="#ffffff",
            button_color="#445344", button_hover_color="#6b9463",
            font=("Arial", 12),
            command=lambda value, key=text: self.selections.update({key: {"method": value, "column": self.selections[key]["column"]}})
        )
        method_menu.pack(pady=(0, 5))

        if self.dataset is not None:
            if text == "Categorical Encoding:":
                col_names = [col for col in self.dataset.columns if self.dataset[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.dataset[col])]
            else:
                col_names = self.dataset.columns.tolist()
            if not col_names:
                col_names = ["No Data"]
        else:
            col_names = ["No Data"]

        column_menu = ctk.CTkOptionMenu(
            frame, values=col_names, fg_color="#445344", text_color="#ffffff",
            button_color="#445344", button_hover_color="#6b9463",
            font=("Arial", 12),
            command=lambda value, key=text: self.selections[key].update({"column": value})
        )
        column_menu.pack(pady=(0, 5))

        apply_button = ctk.CTkButton(
            frame, text="Apply", width=120, height=30,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            font=("Arial", 12, "bold"), command=lambda key=text: self.apply_section_preprocessing(key)
        )
        apply_button.pack(pady=(5, 15))

        return frame

    def show_raw_dataset(self):
        self.load_dataset()
        for widget in self.data_display_frame.winfo_children():
            widget.destroy()

        if self.dataset is None:
            ctk.CTkLabel(
                self.data_display_frame, text="No dataset loaded. Please upload a dataset.",
                font=("Arial", 14), text_color="#cccccc", wraplength=300
            ).pack(pady=(10, 10))
        else:
            top = tk.Toplevel()
            frame = tk.Frame(top)
            frame.pack(fill="both", expand=True)

            self.data_table = Table(
                frame, dataframe=self.dataset.head(5),
                showtoolbar=False, showstatusbar=False, width=600, height=150,
                cellwidth=100, rowheight=25, max_rows=5
            )
            self.data_table.show()

    def show_preprocessed_dataset(self):
        for widget in self.data_display_frame.winfo_children():
            widget.destroy()
        
        if self.preprocessed_dataset is None:
            ctk.CTkLabel(
                self.data_display_frame, text="No preprocessed dataset available. Please apply preprocessing.",
                font=("Arial", 14), text_color="#cccccc", wraplength=300
            ).pack(pady=(10, 10))
        else:
            self.data_table = Table(
                self.data_display_frame, dataframe=self.preprocessed_dataset.head(5),
                showtoolbar=False, showstatusbar=False, width=600, height=150,
                cellwidth=100, rowheight=25, max_rows=5
            )
            self.data_table.show()
            self.data_table.redraw()

    def update_column_menus(self):
        if self.dataset is None:
            return

        for card in self.processing_cards:
            widgets = card.winfo_children()
            label_widget = next(w for w in widgets if isinstance(w, ctk.CTkLabel))
            key = label_widget.cget("text")
            if key == "Categorical Encoding:":
                col_names = [col for col in self.dataset.columns if self.dataset[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.dataset[col])]
            else:
                col_names = self.dataset.columns.tolist()
            if not col_names:
                col_names = ["No Data"]

            for widget in widgets:
                if isinstance(widget, ctk.CTkOptionMenu) and widget.get() in ["No Data"]:
                    widget.configure(values=col_names)
                    widget.set(col_names[0])
                    self.selections[key]["column"] = col_names[0]

    def apply_section_preprocessing(self, key):
        self.load_dataset()
        if self.dataset is None:
            for widget in self.data_display_frame.winfo_children():
                widget.destroy()
            ctk.CTkLabel(
                self.data_display_frame, text="No dataset loaded. Please upload a dataset.",
                font=("Arial", 14), text_color="#cccccc", wraplength=300
            ).pack(pady=(10, 10))
            return
        
        df = self.preprocessed_dataset.copy() if self.preprocessed_dataset is not None else self.dataset.copy()
        
        selection = self.selections.get(key)
        method = selection.get("method")
        column = selection.get("column")

        if key == "Missing Values:":
            if method == "Categorical (Mode)" and column:
                if pd.api.types.is_numeric_dtype(df[column]):
                    print(f"Error: {column} is numerical, please select a categorical column for mode imputation.")
                    return
                df[column] = df[column].fillna(df[column].mode()[0])
            elif method == "Numerical (Drop)":
                df = df.dropna()
            elif method == "Numerical (Fill)" and column:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    print(f"Error: {column} is categorical, please select a numerical column for fill imputation.")
                    return
                df[column] = df[column].fillna(df[column].mean())
            elif method == "Numerical (Simple Imputer)" and column:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    print(f"Error: {column} is categorical, please select a numerical column for simple imputer.")
                    return
                imputer = SimpleImputer(strategy="mean")
                df[[column]] = imputer.fit_transform(df[[column]])

        elif key == "Categorical Encoding:":
            if method == "One-Hot" and column:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(df[[column]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
                df = pd.concat([df.drop(column, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            elif method == "Label Encoding" and column:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])

        elif key == "Normalization:":
            if method == "Min-Max" and column:
                scaler = MinMaxScaler()
                df[[column]] = scaler.fit_transform(df[[column]])
            elif method == "Z-score" and column:
                scaler = StandardScaler()
                df[[column]] = scaler.fit_transform(df[[column]])

        elif key == "Outliers:":
            if method == "Remove" and column:
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                df = df[(df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)]

        self.preprocessed_dataset = df
        self.main_app.shared_data["preprocessed_dataset"] = df
        print(f"Preprocessing for {key} applied successfully")
        self.show_preprocessed_dataset()

    def apply_preprocessing(self):
        self.load_dataset()
        if self.dataset is None:
            for widget in self.data_display_frame.winfo_children():
                widget.destroy()
            ctk.CTkLabel(
                self.data_display_frame, text="No dataset loaded. Please upload a dataset.",
                font=("Arial", 14), text_color="#cccccc", wraplength=300
            ).pack(pady=(10, 10))
            return
        df = self.dataset.copy()
        
        for key, selection in self.selections.items():
            method = selection.get("method")
            column = selection.get("column")

            if key == "Missing Values:":
                if method == "Categorical (Mode)" and column:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        print(f"Error: {column} is numerical, please select a categorical column for mode imputation.")
                        continue
                    df[column] = df[column].fillna(df[column].mode()[0])
                elif method == "Numerical (Drop)":
                    df = df.dropna()
                elif method == "Numerical (Fill)" and column:
                    if not pd.api.types.is_numeric_dtype(df[column]):
                        print(f"Error: {column} is categorical, please select a numerical column for fill imputation.")
                        continue
                    df[column] = df[column].fillna(df[column].mean())
                elif method == "Numerical (Simple Imputer)" and column:
                    if not pd.api.types.is_numeric_dtype(df[column]):
                        print(f"Error: {column} is categorical, please select a numerical column for simple imputer.")
                        continue
                    imputer = SimpleImputer(strategy="mean")
                    df[[column]] = imputer.fit_transform(df[[column]])

            elif key == "Categorical Encoding:":
                if method == "One-Hot" and column:
                    encoder = OneHotEncoder(sparse_output=False, drop='first')
                    encoded = encoder.fit_transform(df[[column]])
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
                    df = pd.concat([df.drop(column, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
                elif method == "Label Encoding" and column:
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])

            elif key == "Normalization:":
                if method == "Min-Max" and column:
                    scaler = MinMaxScaler()
                    df[[column]] = scaler.fit_transform(df[[column]])
                elif method == "Z-score" and column:
                    scaler = StandardScaler()
                    df[[column]] = scaler.fit_transform(df[[column]])

            elif key == "Outliers:":
                if method == "Remove" and column:
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    df = df[(df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)]

        self.preprocessed_dataset = df
        self.main_app.shared_data["preprocessed_dataset"] = df
        print("All preprocessing steps applied successfully")
        self.show_preprocessed_dataset()