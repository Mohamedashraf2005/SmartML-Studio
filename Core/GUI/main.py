import customtkinter as ctk
from tkinter import ttk
from page1 import DataUploaderApp
from page2 import DataDashboardApp
from page3 import DataPreprocessingApp
from page4 import ModelSelectionApp
 
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis App")
        self.root.geometry("1000x600")
        self.root.minsize(800, 400)
        self.root.configure(bg="#000000")

        self.style = {
            "nav_fg_color": "#435242",
            "btn_fg_color": "#445344",
            "btn_active_color": "#5e7a51",
            "btn_disabled_color": "#666666",
            "btn_hover_color": "#6b9463",
            "btn_border_color": "#5e7a51",
            "text_color": "#ffffff",
            "font": ("Arial", 16, "bold"),
            "font_small": ("Arial", 14),
            "btn_width": 180,
            "btn_height": 40,
            "corner_radius": 10
        }

        self.pages = {}
        self.shared_data = {}
        self.current_page = None
        self.create_nav()
        self.create_container()
        self.load_pages()
        self.show_page("Uploader")

    def create_nav(self):
        self.nav_frame = ctk.CTkFrame(self.root, height=60, fg_color=self.style["nav_fg_color"])
        self.nav_frame.pack(side="top", fill="x")

        nav_container = ctk.CTkFrame(self.nav_frame, fg_color="transparent")
        nav_container.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            nav_container, text="Data Analyzer", font=("Arial", 20, "bold"), text_color="#ffffff"
        ).pack(side="left", padx=(0, 20))

        buttons = [
            ("Data Uploader", "Uploader", "normal"),
            ("Dashboard", "Dashboard", "disabled"),
            ("Preprocessing", "Preprocessing", "disabled"),
            ("Model Selection", "ModelSelection", "disabled")
        ]

        self.nav_buttons = {}
        for text, page, state in buttons:
            btn = ctk.CTkButton(
                nav_container, text=text, command=lambda p=page: self.show_page(p),
                fg_color=self.style["btn_fg_color"], text_color=self.style["text_color"],
                hover_color=self.style["btn_hover_color"], border_width=2,
                border_color=self.style["btn_border_color"], corner_radius=self.style["corner_radius"],
                width=self.style["btn_width"], height=self.style["btn_height"],
                font=self.style["font"], state=state
            )
            btn.pack(side="left", padx=15, pady=5)
            self.nav_buttons[page] = btn
            if text != buttons[-1][0]:
                ctk.CTkLabel(nav_container, text="|", font=("Arial", 16), text_color="#5e7a51").pack(side="left")

    def create_container(self):
        self.container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.container.pack(fill="both", expand=True)

    def load_pages(self):
        try:
            uploader_frame = ctk.CTkFrame(self.container)
            dashboard_frame = ctk.CTkFrame(self.container)
            preprocessing_frame = ctk.CTkFrame(self.container)
            model_selection_frame = ctk.CTkFrame(self.container)

            self.pages["Uploader"] = uploader_frame
            self.pages["Dashboard"] = dashboard_frame
            self.pages["Preprocessing"] = preprocessing_frame
            self.pages["ModelSelection"] = model_selection_frame

            self.uploader_app = DataUploaderApp(uploader_frame, self)
            self.dashboard_app = DataDashboardApp(dashboard_frame, self)
            self.preprocessing_app = DataPreprocessingApp(preprocessing_frame, self)
            self.model_selection_app = ModelSelectionApp(model_selection_frame, self)
        except Exception as e:
            print(f"Error loading pages: {e}")

    def show_page(self, name):
        if name not in self.pages:
            print(f"Error: Page '{name}' not found.")
            return
        if self.current_page:
            self.current_page.pack_forget()
        self.current_page = self.pages[name]
        self.current_page.pack(fill="both", expand=True)
        # Trigger page-specific initialization with data check
        if name == "Uploader":
            pass
        elif name == "Dashboard" and self.dashboard_app:
            self.dashboard_app.load_dataset()
            self.dashboard_app.show_card(self.dashboard_app.tab_names[0])
        elif name == "Preprocessing" and self.preprocessing_app:
            self.preprocessing_app.load_dataset()
            self.preprocessing_app.update_column_menus()
        elif name == "ModelSelection" and self.model_selection_app:
            self.model_selection_app.load_dataset()
            self.model_selection_app.update_ui_state()

    def enable_nav_buttons(self):
        for btn in self.nav_buttons.values():
            btn.configure(state="normal")

if __name__ == "__main__":
    root = ctk.CTk()
    app = MainApp(root)
    root.mainloop()