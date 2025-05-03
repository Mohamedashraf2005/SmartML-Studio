import customtkinter as ctk
import tkinter as tk

class DataDashboardApp:
    def __init__(self, parent):
        self.root = parent
        self.root.configure(fg_color="#435242")

        self.button_width = 180
        self.button_height = 40
        self.button_font = ("Arial", 14)
        self.button_pad_y = 10
        self.button_pad_x = 15
        self.button_color = "#435242"
        self.button_text_color = 'white'
        self.button_hover_color = "#5e7a51"
        self.border_color="#5e7a51"
        self.border_width=2
        self.corner_radius=12

        self.tab_names = [
            "Dataset summary, Structure", "Ydata porfiling", "Show hist",
            "Show heatmap", "Show pairplot", "Show Barchart", "Generate Your own plot"
        ]

        self.frames = {}
        self.setup_layout()

    def setup_layout(self):
        self.create_sidebar()
        self.create_content_area()
        self.show_frame(self.tab_names[0])

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self.root, width=200, fg_color='#435242')
        self.sidebar.pack(side="left", fill="y", padx=5, pady=10)

        for tab_name in self.tab_names:
            button = ctk.CTkButton(
                self.sidebar,
                text=tab_name,
                width=self.button_width,
                height=self.button_height,
                font=self.button_font,
                fg_color=self.button_color,
                text_color=self.button_text_color,
                hover_color=self.button_hover_color,
                border_color=self.border_color,
                border_width=self.border_width,
                corner_radius=self.corner_radius,
                command=lambda name=tab_name: self.show_frame(name)
            )
            button.pack(pady=self.button_pad_y, padx=self.button_pad_x, fill="x")

        next_page_button = ctk.CTkButton(
            self.sidebar,
            text="Next Page",
            width=self.button_width,
            height=self.button_height,
            font=self.button_font,
            fg_color=self.button_color,
            text_color=self.button_text_color,
            hover_color=self.button_hover_color,
            command=self.go_to_next_page
        )
        next_page_button.pack(side="bottom", pady=10, padx=15)

    def create_content_area(self):
        self.content_area = ctk.CTkFrame(self.root, fg_color='#435242', corner_radius=10)
        self.content_area.pack(side="right", expand=True, fill="both", pady=10, padx=10)

        for tab_name in self.tab_names:
            frame = ctk.CTkFrame(self.content_area, fg_color='#435242', corner_radius=10)
            label = ctk.CTkLabel(frame, text=f"Content for: {tab_name}", font=("Arial", 20), text_color="white")
            label.pack(pady=20)
            self.frames[tab_name] = frame

    def show_frame(self, name):
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[name].pack(expand=True, fill="both", padx=10, pady=10)

    def go_to_next_page(self):
        # Navigate to the Preprocessing page by calling show_page from MainApp
        main_app = self.root.master.master  # Access MainApp instance
        main_app.show_page("Preprocessing")
