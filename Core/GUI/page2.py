import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np

class DataDashboardApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.dataset = None
        self.root.configure(fg_color="#435242")

        self.button_width = 180
        self.button_height = 40
        self.button_font = ("Arial", 14, "bold")
        self.button_pad_y = 5
        self.button_pad_x = 10
        self.button_color = "#445344"
        self.button_active_color = "#5e7a51"
        self.button_text_color = "#ffffff"
        self.button_hover_color = "#6b9463"
        self.border_color = "#5e7a51"
        self.border_width = 2
        self.corner_radius = 10

        self.tab_names = [
            "Dataset Summary", "Show Dataset", "Show Histogram",
            "Show Heatmap", "Show Pairplot", "Show Barchart", "Generate Custom Plot"
        ]
        self.active_tab = self.tab_names[0]

        self.chart_frames = []
        self.image_chart_frames = []
        self.current_chart_index = 0
        self.image_chart_index = 0

        self.setup_layout()

    def load_dataset(self):
        try:
            file_path = self.main_app.shared_data.get("dataset_path")
            if file_path:
                if file_path.endswith(".csv"):
                    self.dataset = pd.read_csv(file_path)
                elif file_path.endswith(".xlsx"):
                    self.dataset = pd.read_excel(file_path)
                print(f"Dataset loaded from path: {file_path} ({self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns)")
            else:
                self.dataset = self.main_app.shared_data.get("preprocessed_dataset")
                if self.dataset is not None:
                    print(f"Preprocessed dataset loaded: ({self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns)")
                else:
                    print("No dataset or preprocessed dataset available in Dashboard")
        except Exception as e:
            print(f"Error loading dataset in Dashboard: {e}")
            self.dataset = None

    def setup_layout(self):
        self.container = ctk.CTkFrame(self.root, fg_color="#435242")
        self.container.pack(expand=True, fill="both", padx=10, pady=10)

        ctk.CTkLabel(
            self.container, text="Dashboard Insights", font=("Arial", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=(10, 5))
        ctk.CTkLabel(
            self.container, text="Explore your dataset with interactive visualizations",
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

        self.tab_buttons = {}
        for idx, tab_name in enumerate(self.tab_names):
            fg_color = self.button_active_color if tab_name == self.active_tab else self.button_color
            button = ctk.CTkButton(
                self.sidebar, text=tab_name, width=self.button_width, height=self.button_height,
                font=self.button_font, fg_color=fg_color, text_color=self.button_text_color,
                hover_color=self.button_hover_color, border_color=self.border_color,
                border_width=self.border_width, corner_radius=self.corner_radius,
                command=lambda name=tab_name: self.show_card(name)
            )
            button.pack(pady=self.button_pad_y, padx=self.button_pad_x, fill="x")
            self.tab_buttons[tab_name] = button
            if idx < len(self.tab_names) - 1:
                ctk.CTkLabel(self.sidebar, text="─" * 30, font=("Arial", 12), text_color="#5e7a51").pack(pady=5)

    def create_content_area(self, parent):
        self.cards_container = ctk.CTkFrame(parent, fg_color="#435242")
        self.cards_container.grid(row=0, column=1, sticky="nsew")
        self.cards_container.grid_rowconfigure(0, weight=1)
        self.cards_container.grid_columnconfigure(0, weight=1)

        self.current_card = ctk.CTkFrame(
            self.cards_container, fg_color="#445344", corner_radius=15,
            border_width=3, border_color="#3a473a"
        )
        self.current_card.place(relwidth=0.95, relheight=0.9)
        self.current_card.bind("<Enter>", lambda e: self.current_card.configure(fg_color="#5e7a51"))
        self.current_card.bind("<Leave>", lambda e: self.current_card.configure(fg_color="#445344"))

        nav_frame = ctk.CTkFrame(self.cards_container, fg_color="transparent")
        nav_frame.place(relx=0.5, rely=0.95, anchor="center")

        back_button = ctk.CTkButton(
            nav_frame, text="Back to Uploader", width=150, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            command=lambda: self.main_app.show_page("Uploader"), font=("Arial", 14)
        )
        back_button.pack(side="left", padx=10)

        next_button = ctk.CTkButton(
            nav_frame, text="Next Page", width=150, height=40,
            fg_color="#445344", text_color="#ffffff", hover_color="#6b9463",
            command=self.go_to_next_page, font=("Arial", 14)
        )
        next_button.pack(side="left", padx=10)

        self.show_card(self.tab_names[0])

    def create_card_content(self, text):
        for widget in self.current_card.winfo_children():
            widget.destroy()

        if self.dataset is None or self.dataset.empty:
            ctk.CTkLabel(
                self.current_card, text="No dataset loaded. Please upload a dataset in the Uploader page.",
                font=("Arial", 16), text_color="#cccccc", wraplength=300
            ).pack(pady=(20, 10))
            return

        if text == "Dataset Summary":
            # Create a scrollable frame for the summary
            scrollable_frame = ctk.CTkScrollableFrame(
                self.current_card,
                width=380,
                height=300,  # Adjust height as needed
                corner_radius=10,
                fg_color="transparent",  # Transparent background
                scrollbar_button_color="#4a4a4a",  # Scrollbar button color
                scrollbar_button_hover_color="#666666"  # Scrollbar hover color
            )
            scrollable_frame.pack(pady=(20, 10), padx=10, fill="both", expand=True)

            # Format the summary text
            summary = f"DataSet Shape \nRows: {self.dataset.shape[0]}, Columns: {self.dataset.shape[1]}\n\n"
            summary += f"Total Null Values: {self.dataset.isnull().sum().sum()}\n"
            summary += f"Duplicate Rows: {self.dataset.duplicated().sum()}\n\n"
            summary += "\nData Types:\n"
            summary += f"{self.dataset.dtypes.to_string()}\n"

            # Add the summary label inside the scrollable frame
            ctk.CTkLabel(
                scrollable_frame,
                text=summary,
                font=("Arial", 14),
                text_color="#ffffff",
                wraplength=350,  # Slightly less than frame width to avoid text cutoff
                justify="left",
                anchor="w"
            ).pack(pady=10, padx=10, fill="x")

        elif text == "Show Dataset":
            # Create a frame for the table and scrollbars
            table_frame = ctk.CTkFrame(self.current_card, fg_color="#445344")
            table_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Create scrollbars
            scrollbar_y = ctk.CTkScrollbar(table_frame, orientation="vertical")
            scrollbar_x = ctk.CTkScrollbar(table_frame, orientation="horizontal")

            # Create the Treeview
            tree = ttk.Treeview(
                table_frame,
                yscrollcommand=scrollbar_y.set,
                xscrollcommand=scrollbar_x.set,
                height=20  # Increased height for better visibility
            )
            scrollbar_y.configure(command=tree.yview)
            scrollbar_x.configure(command=tree.xview)

            # Place the tree and scrollbars using grid
            tree.grid(row=0, column=0, sticky="nsew")
            scrollbar_y.grid(row=0, column=1, sticky="ns")
            scrollbar_x.grid(row=1, column=0, sticky="ew")

            # Configure grid weights
            table_frame.grid_rowconfigure(0, weight=1)
            table_frame.grid_columnconfigure(0, weight=1)

            # Define columns
            columns = list(self.dataset.columns)
            tree["columns"] = columns
            tree.heading("#0", text="Index", anchor="center")
            for col in columns:
                tree.heading(col, text=col, anchor="center", command=lambda c=col: self.sort_column(tree, c, False))
                # Dynamically adjust column width based on content
                max_width = max(
                    len(str(col)) * 10,  # Based on header
                    max([len(str(x)) for x in self.dataset[col].dropna()] + [0]) * 8, 100  # Based on data
                )
                tree.column(col, width=min(max_width, 200), anchor="center", stretch=True)

            # Set index column width
            tree.column("#0", width=80, anchor="center")

            # Insert data
            for idx, row in self.dataset.iterrows():
                values = [str(row[col]) if pd.notna(row[col]) else "N/A" for col in columns]
                tree.insert("", "end", text=str(idx), values=values, tags=("evenrow" if idx % 2 == 0 else "oddrow",))

            # Style the Treeview
            style = ttk.Style()
            style.configure(
                "Treeview",
                background="#445344",
                foreground="#ffffff",
                fieldbackground="#445344",
                font=("Arial", 11),
                rowheight=30
            )
            style.configure(
                "Treeview.Heading",
                background="#5e7a51",
                foreground="#ffffff",
                font=("Arial", 12, "bold"),
                anchor="center"
            )
            style.map(
                "Treeview.Heading",
                background=[("active", "#6b9463")]
            )
            tree.tag_configure("oddrow", background="#3a473a")
            tree.tag_configure("evenrow", background="#445344")

            # Store tree for sorting
            self.tree = tree

        elif text == "Show Histogram":
            main_frame = ctk.CTkFrame(self.current_card, fg_color="#445344")
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            control_frame = ctk.CTkFrame(main_frame, fg_color="#445344")
            control_frame.pack(side="left", padx=10, pady=10, fill="y")

            self.histogram_frame = ctk.CTkFrame(main_frame, fg_color="#445344")
            self.histogram_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

            ctk.CTkLabel(control_frame, text="Select Columns (up to 6)", font=("Arial", 14)).pack(pady=5)
            self.hist_feature_listbox = tk.Listbox(control_frame, selectmode="multiple", height=10, width=30)
            self.hist_feature_listbox.pack(pady=5)

            if self.dataset is not None:
                for col in self.dataset.columns:
                    self.hist_feature_listbox.insert("end", col)

            ctk.CTkButton(
                control_frame, text="Generate Histograms", fg_color="#00b300",
                command=self.generate_histograms
            ).pack(pady=10)

        elif text == "Show Heatmap":
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(self.dataset.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.current_card)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, padx=10, fill="both", expand=True)

        elif text == "Show Pairplot":
            main_frame = ctk.CTkFrame(self.current_card, fg_color="#445344")
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            control_frame = ctk.CTkFrame(main_frame, fg_color="#435242")
            control_frame.pack(side="left", padx=10, pady=10, fill="y")

            self.pairplot_frame = ctk.CTkFrame(main_frame, fg_color="#435242")
            self.pairplot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

            ctk.CTkLabel(control_frame, text="Select Columns (up to 6)", font=("Arial", 14)).pack(pady=5)
            self.feature_listbox = tk.Listbox(control_frame, selectmode="multiple", height=10, width=30)
            self.feature_listbox.pack(pady=5)

            if self.dataset is not None:
                for col in self.dataset.columns:
                    self.feature_listbox.insert("end", col)

            ctk.CTkButton(
                control_frame, text="Generate Pairplot", fg_color="#00b300",
                command=self.generate_pairplot
            ).pack(pady=10)

        elif text == "Show Barchart":
            main_frame = ctk.CTkFrame(self.current_card, fg_color="#445344")
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            control_frame = ctk.CTkFrame(main_frame, fg_color="#445344")
            control_frame.pack(side="left", padx=10, pady=10, fill="y")

            self.barchart_frame = ctk.CTkFrame(main_frame, fg_color="#445344")
            self.barchart_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

            ctk.CTkLabel(control_frame, text="Select Columns (up to 6)", font=("Arial", 14)).pack(pady=5)
            self.barchart_feature_listbox = tk.Listbox(control_frame, selectmode="multiple", height=10, width=30)
            self.barchart_feature_listbox.pack(pady=5)

            if self.dataset is not None:
                # Prefer categorical or discrete columns for bar charts
                for col in self.dataset.columns:
                    self.barchart_feature_listbox.insert("end", col)

            ctk.CTkButton(
                control_frame, text="Generate Bar Charts", fg_color="#00b300",
                command=self.generate_barcharts
            ).pack(pady=10)

        elif text == "Generate Custom Plot":
            main_frame = ctk.CTkFrame(self.current_card, fg_color="#445344")
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            control_frame = ctk.CTkFrame(main_frame, fg_color="#445344")
            control_frame.pack(side="left", padx=10, pady=10, fill="y")

            plot_frame = ctk.CTkFrame(main_frame, fg_color="#445344")
            plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

            self.image_chart_frames = [ctk.CTkFrame(plot_frame, fg_color="#445344")]
            self.image_chart_frames[0].pack(fill="both", expand=True, padx=5, pady=5)

            self.chart_dropdown_var = ctk.StringVar(value="Histogram")
            self.chart_dropdown = ctk.CTkOptionMenu(
                control_frame, values=["Bar Chart", "Histogram", "Box plot", "Pie Chart"],
                variable=self.chart_dropdown_var, command=self.toggle_column_dropdown
            )
            self.chart_dropdown.pack(pady=(20, 5))

            column_values = self.dataset.columns.tolist() if self.dataset is not None else []
            self.hist_column_label = ctk.CTkLabel(control_frame, text="Select Column")
            self.hist_column_label.pack(pady=(5, 5))
            self.hist_column_menu = ctk.CTkOptionMenu(control_frame, values=column_values)
            self.hist_column_menu.pack(pady=(0, 10))

            self.bar_column_label = ctk.CTkLabel(control_frame, text="Select Column for Bar Chart")
            self.bar_column_menu = ctk.CTkOptionMenu(control_frame, values=column_values)

            self.boxplot_column_label = ctk.CTkLabel(control_frame, text="Select Column for boxplot Chart")
            self.boxplot_column_menu = ctk.CTkOptionMenu(control_frame, values=column_values)

            self.pie_column_label = ctk.CTkLabel(control_frame, text="Select Column for Pie Chart")
            self.pie_column_menu = ctk.CTkOptionMenu(control_frame, values=column_values)

            self.generate_chart_button = ctk.CTkButton(
                control_frame, text="Generate Chart", fg_color="#00b300", command=self.generate_chart
            )
            self.generate_chart_button.pack(pady=(10, 20))

            self.reset_plots_button = ctk.CTkButton(
                control_frame, text="Reset Charts", fg_color="Red", command=self.reset_plots
            )
            self.reset_plots_button.pack(pady=(10, 30))

            self.toggle_column_dropdown("Histogram")
    def generate_barcharts(self):
        if self.dataset is None or self.dataset.empty:
            ctk.CTkLabel(
                self.barchart_frame, text="No dataset loaded. Please upload a dataset.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        selected_indices = self.barchart_feature_listbox.curselection()
        selected_columns = [self.barchart_feature_listbox.get(i) for i in selected_indices]

        if len(selected_columns) > 6:
            ctk.CTkLabel(
                self.barchart_frame, text="Please select up to 6 columns.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        if not selected_columns:
            ctk.CTkLabel(
                self.barchart_frame, text="Please select at least one column.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        for widget in self.barchart_frame.winfo_children():
            widget.destroy()

        try:
            n_cols = len(selected_columns)
            n_rows = int(np.ceil(n_cols / 2))
            if n_cols == 1:
                n_rows, n_cols = 1, 1
            elif n_cols <= 2:
                n_rows, n_cols = 1, n_cols
            else:
                n_cols = 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
            axes = np.array(axes).ravel()

            for i, col in enumerate(selected_columns):
                # Use value_counts for bar chart
                value_counts = self.dataset[col].value_counts().head(10)  # Limit to top 10 categories for readability
                value_counts.plot(kind='bar', ax=axes[i], color='orange', edgecolor='black')
                axes[i].set_title(f'Bar Chart of {col}', fontsize=10)
                axes[i].set_xlabel(col, fontsize=8)
                axes[i].set_ylabel("Count", fontsize=8)
                axes[i].set_xticks([])  # Disable x-axis ticks and labels
                axes[i].set_yticks([])  # Disable y-axis ticks and labels

            for i in range(len(selected_columns), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.barchart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        except Exception as e:
            ctk.CTkLabel(
                self.barchart_frame, text=f"Error generating bar charts: {e}",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)

    # def generate_heatmap(self):
    #     if self.dataset is None or self.dataset.empty:
    #         ctk.CTkLabel(
    #             self.heatmap_frame, text="No dataset loaded. Please upload a dataset.",
    #             font=("Arial", 14), text_color="#ff0000"
    #         ).pack(pady=10)
    #         return

    #     selected_indices = self.heatmap_feature_listbox.curselection()
    #     selected_columns = [self.heatmap_feature_listbox.get(i) for i in selected_indices]

    #     if len(selected_columns) > 10:
    #         ctk.CTkLabel(
    #             self.heatmap_frame, text="Please select up to 10 columns.",
    #             font=("Arial", 14), text_color="#ff0000"
    #         ).pack(pady=10)
    #         return

    #     if not selected_columns:
    #         ctk.CTkLabel(
    #             self.heatmap_frame, text="Please select at least one column.",
    #             font=("Arial", 14), text_color="#ff0000"
    #         ).pack(pady=10)
    #         return

    #     for widget in self.heatmap_frame.winfo_children():
    #         widget.destroy()

    #     numeric_columns = [col for col in selected_columns if self.dataset[col].dtype in ['float64', 'int64']]
    #     if not numeric_columns:
    #         ctk.CTkLabel(
    #             self.heatmap_frame, text="Selected columns must be numeric.",
    #             font=("Arial", 14), text_color="#ff0000"
    #         ).pack(pady=10)
    #         return

    #     try:
    #         corr_matrix = self.dataset[numeric_columns].corr()
    #         fig, ax = plt.subplots(figsize=(6, 4))
    #         sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax, fmt=".2f", vmin=-1, vmax=1)
    #         ax.set_title("Correlation Heatmap", fontsize=12)
    #         plt.tight_layout()

    #         canvas = FigureCanvasTkAgg(fig, master=self.heatmap_frame)
    #         canvas.draw()
    #         canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    #     except Exception as e:
    #         ctk.CTkLabel(
    #             self.heatmap_frame, text=f"Error generating heatmap: {e}",
    #             font=("Arial", 14), text_color="#ff0000"
    #         ).pack(pady=10)

    def generate_pairplot(self):
        if self.dataset is None or self.dataset.empty:
            ctk.CTkLabel(
                self.pairplot_frame, text="No dataset loaded. Please upload a dataset.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        selected_indices = self.feature_listbox.curselection()
        selected_columns = [self.feature_listbox.get(i) for i in selected_indices]

        if len(selected_columns) > 6:
            ctk.CTkLabel(
                self.pairplot_frame, text="Please select up to 6 columns.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        if not selected_columns:
            ctk.CTkLabel(
                self.pairplot_frame, text="Please select at least one column.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        for widget in self.pairplot_frame.winfo_children():
            widget.destroy()

        numeric_columns = [col for col in selected_columns if self.dataset[col].dtype in ['float64', 'int64']]
        if not numeric_columns:
            ctk.CTkLabel(
                self.pairplot_frame, text="Selected columns must be numeric.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        try:
            pair_plot = sns.pairplot(self.dataset[numeric_columns])
            fig = pair_plot.figure
            fig.set_size_inches(6, 6)
            canvas = FigureCanvasTkAgg(fig, master=self.pairplot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        except Exception as e:
            ctk.CTkLabel(
                self.pairplot_frame, text=f"Error generating pairplot: {e}",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)

    def generate_histograms(self):
        if self.dataset is None or self.dataset.empty:
            ctk.CTkLabel(
                self.histogram_frame, text="No dataset loaded. Please upload a dataset.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        selected_indices = self.hist_feature_listbox.curselection()
        selected_columns = [self.hist_feature_listbox.get(i) for i in selected_indices]

        if len(selected_columns) > 6:
            ctk.CTkLabel(
                self.histogram_frame, text="Please select up to 6 columns.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        if not selected_columns:
            ctk.CTkLabel(
                self.histogram_frame, text="Please select at least one column.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        for widget in self.histogram_frame.winfo_children():
            widget.destroy()

        numeric_columns = [col for col in selected_columns if self.dataset[col].dtype in ['float64', 'int64']]
        if not numeric_columns:
            ctk.CTkLabel(
                self.histogram_frame, text="Selected columns must be numeric.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        try:
            n_cols = len(numeric_columns)
            n_rows = int(np.ceil(n_cols / 2))
            if n_cols == 1:
                n_rows, n_cols = 1, 1
            elif n_cols <= 2:
                n_rows, n_cols = 1, n_cols
            else:
                n_cols = 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
            axes = np.array(axes).ravel()

            for i, col in enumerate(numeric_columns):
                self.dataset[col].hist(ax=axes[i], bins=20)
                axes[i].set_title(col)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Count")

            for i in range(len(numeric_columns), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        except Exception as e:
            ctk.CTkLabel(
                self.histogram_frame, text=f"Error generating histograms: {e}",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)

    def toggle_column_dropdown(self, choice):
        self.hist_column_label.pack_forget()
        self.hist_column_menu.pack_forget()
        self.bar_column_label.pack_forget()
        self.bar_column_menu.pack_forget()
        self.boxplot_column_label.pack_forget()
        self.boxplot_column_menu.pack_forget()
        self.pie_column_label.pack_forget()
        self.pie_column_menu.pack_forget()

        column_values = self.dataset.columns.tolist() if self.dataset is not None else []
        self.hist_column_menu.configure(values=column_values)
        self.bar_column_menu.configure(values=column_values)
        self.boxplot_column_menu.configure(values=column_values)
        self.pie_column_menu.configure(values=column_values)

        if choice == "Histogram":
            self.hist_column_label.pack(pady=(5, 5))
            self.hist_column_menu.pack(pady=(0, 10))
        elif choice == "Bar Chart":
            self.bar_column_label.pack(pady=(5, 5))
            self.bar_column_menu.pack(pady=(0, 10))
        elif choice == "Box plot":
            self.boxplot_column_label.pack(pady=(5, 5))
            self.boxplot_column_menu.pack(pady=(0, 10))
        elif choice == "Pie Chart":
            self.pie_column_label.pack(pady=(5, 5))
            self.pie_column_menu.pack(pady=(0, 10))

    def generate_chart(self):
        if self.dataset is None or self.dataset.empty:
            ctk.CTkLabel(
                self.image_chart_frames[0], text="No dataset loaded. Please upload a dataset.",
                font=("Arial", 14), text_color="#ff0000"
            ).pack(pady=10)
            return

        chart_type = self.chart_dropdown_var.get()
        if chart_type == "Histogram":
            self.plot_image_histogram()
        elif chart_type == "Bar Chart":
            self.plot_image_bar_chart()
        elif chart_type == "Box plot":
            self.plot_image_boxplot_plot()
        elif chart_type == "Pie Chart":
            self.plot_image_pie_chart()

    def plot_image_histogram(self):
        column = self.hist_column_menu.get()
        if column not in self.dataset.columns:
            return
        fig, ax = plt.subplots(figsize=(4, 3))
        self.dataset[column].hist(ax=ax, color='lightgreen', edgecolor='black')
        ax.set_title(f'Histogram of {column}', fontsize=10)
        plt.tight_layout()
        self.display_image_plot(fig)

    def plot_image_bar_chart(self):
        column = self.bar_column_menu.get()
        if column not in self.dataset.columns:
            return
        fig, ax = plt.subplots(figsize=(4, 3))
        self.dataset[column].value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
        ax.set_title(f'Bar Chart of {column}', fontsize=10)
        plt.tight_layout()
        self.display_image_plot(fig)

    def plot_image_boxplot_plot(self):
        column = self.boxplot_column_menu.get()
        if column not in self.dataset.columns:
            return
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.boxplot(self.dataset[column].dropna(), patch_artist=True,
                   boxprops=dict(facecolor='purple', color='black'),
                   medianprops=dict(color='yellow'))
        ax.set_title(f'Box Plot of {column}', fontsize=10)
        ax.set_ylabel(column, fontsize=8)
        plt.tight_layout()
        self.display_image_plot(fig)

    def plot_image_pie_chart(self):
        column = self.pie_column_menu.get()
        if column not in self.dataset.columns:
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        self.dataset[column].value_counts().plot(kind='pie', ax=ax, colors=['lightcoral', 'lightblue', 'lightgreen', 'orange'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Pie Chart of {column}', fontsize=10)
        plt.tight_layout()
        self.display_image_plot(fig)

    def display_image_plot(self, fig):
        if self.image_chart_index >= len(self.image_chart_frames):
            return

        chart_frame = self.image_chart_frames[self.image_chart_index]
        for widget in chart_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=False, padx=5, pady=5)
        self.image_chart_index += 1

    def reset_plots(self):
        self.current_chart_index = 0
        self.image_chart_index = 0

        for frame in self.chart_frames:
            for widget in frame.winfo_children():
                widget.destroy()

        for frame in self.image_chart_frames:
            for widget in frame.winfo_children():
                widget.destroy()

    def show_card(self, name):
        self.load_dataset()
        self.active_tab = name
        for tab_name, button in self.tab_buttons.items():
            button.configure(fg_color=self.button_active_color if tab_name == name else self.button_color)
        self.create_card_content(name)

    def go_to_next_page(self):
        self.main_app.show_page("Preprocessing")