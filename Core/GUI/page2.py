import customtkinter as ctk
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class DataDashboardApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.dataset = None
        self.load_dataset()
        self.root.configure(fg_color="#435242")

        self.button_width = 180  # Reduced width for smaller buttons
        self.button_height = 40  # Reduced height for smaller buttons
        self.button_font = ("Arial", 14, "bold")  # Smaller font size
        self.button_pad_y = 5  # Reduced padding for tighter spacing
        self.button_pad_x = 10
        self.button_color = "#445344"
        self.button_active_color = "#5e7a51"
        self.button_text_color = "#ffffff"
        self.button_hover_color = "#6b9463"
        self.border_color = "#5e7a51"
        self.border_width = 2
        self.corner_radius = 10

        self.tab_names = [
            "Dataset summary, Structure", "Ydata profiling", "Show hist",
            "Show heatmap", "Show pairplot", "Show Barchart", "Generate Your own plot"
        ]
        self.active_tab = self.tab_names[0]

        self.setup_layout()

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
        if self.dataset is None:
            ctk.CTkLabel(
                self.current_card, text="No data available. Please upload a dataset.",
                font=("Arial", 16), text_color="#cccccc", wraplength=300
            ).pack(pady=(20, 10))
            return
        if text == "Show hist":
            fig, ax = plt.subplots(figsize=(4, 3))
            self.dataset.hist(ax=ax, bins=20)
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.current_card)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, padx=10, fill="both", expand=True)
        elif text == "Show heatmap":
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(self.dataset.corr(), annot=True, cmap="coolwarm", ax=ax)
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.current_card)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, padx=10, fill="both", expand=True)
        else:
            ctk.CTkLabel(
                self.current_card, text=f"Content for: {text}", font=("Arial", 16),
                text_color="#ffffff", wraplength=300
            ).pack(pady=(20, 10))

    def show_card(self, name):
        for widget in self.current_card.winfo_children():
            widget.destroy()
        self.active_tab = name
        for tab_name, button in self.tab_buttons.items():
            button.configure(fg_color=self.button_active_color if tab_name == name else self.button_color)
        self.create_card_content(name)

    def go_to_next_page(self):
        self.main_app.show_page("Preprocessing")
