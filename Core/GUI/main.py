import customtkinter as ctk
from tkinter import ttk
from page1 import DataUploaderApp
from page2 import DataDashboardApp
from page3 import DataPreprocessingApp  # Import the new page

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")
        self.root.geometry("1000x600")
        self.root.configure(fg_color="#435242")

        self.pages = {}

        self.create_nav()
        self.create_container()
        self.load_pages()
        self.show_page("Uploader")

    def create_nav(self):
        self.nav_frame = ctk.CTkFrame(self.root, height=50, fg_color="#435242")
        self.nav_frame.pack(side="top", fill="x")

        uploader_btn = ctk.CTkButton(
            self.nav_frame, text="Data Uploader",
            command=lambda: self.show_page("Uploader"),
            fg_color="#445344", text_color="white", hover_color="#5e7a51",
            border_width=2, border_color="#5e7a51", corner_radius=10,
            width=180, height=40, font=("Arial", 16, "bold")
        )
        uploader_btn.pack(side="left", padx=(150, 20), pady=10)

        dashboard_btn = ctk.CTkButton(
            self.nav_frame, text="Dashboard",
            command=lambda: self.show_page("Dashboard"),
            fg_color="#445344", text_color="white", hover_color="#5e7a51",
            border_width=2, border_color="#5e7a51", corner_radius=10,
            width=180, height=40, font=("Arial", 16, "bold")
        )
        dashboard_btn.pack(side="left", padx=10, pady=10)

        # Add Preprocessing button
        preprocessing_btn = ctk.CTkButton(
            self.nav_frame, text="Preprocessing",
            command=lambda: self.show_page("Preprocessing"),
            fg_color="#445344", text_color="white", hover_color="#5e7a51",
            border_width=2, border_color="#5e7a51", corner_radius=10,
            width=180, height=40, font=("Arial", 16, "bold")
        )
        preprocessing_btn.pack(side="left", padx=10, pady=10)

    def create_container(self):
        self.container = ctk.CTkFrame(self.root, fg_color="#445344")
        self.container.pack(fill="both", expand=True)

    def load_pages(self):
        uploader_frame = ctk.CTkFrame(self.container)
        dashboard_frame = ctk.CTkFrame(self.container)
        preprocessing_frame = ctk.CTkFrame(self.container)  # Add Preprocessing frame

        uploader_frame.pack(fill="both", expand=True)
        dashboard_frame.pack(fill="both", expand=True)
        preprocessing_frame.pack(fill="both", expand=True)

        self.pages["Uploader"] = uploader_frame
        self.pages["Dashboard"] = dashboard_frame
        self.pages["Preprocessing"] = preprocessing_frame  # Add to pages dict

        self.uploader_app = DataUploaderApp(uploader_frame)
        self.dashboard_app = DataDashboardApp(dashboard_frame)
        self.preprocessing_app = DataPreprocessingApp(preprocessing_frame)  # Initialize the new page

    def show_page(self, name):
        for frame in self.pages.values():
            frame.pack_forget()
        self.pages[name].pack(fill="both", expand=True)

if __name__ == "__main__":
    root = ctk.CTk()
    app = MainApp(root)
    root.mainloop()
