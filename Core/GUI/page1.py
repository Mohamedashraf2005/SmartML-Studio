import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

class DataUploaderApp:
    def __init__(self, parent, main_app):
        self.root = parent
        self.main_app = main_app
        self.root.configure(fg_color="#435242")
        self.create_widgets()

    def create_widgets(self):
        self.frame = ctk.CTkFrame(
            master=self.root, fg_color="#435242", corner_radius=20,
            border_width=3, border_color="#3a473a"
        )
        self.frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.5)
        self.frame.pack_propagate(False)

        ctk.CTkLabel(
            self.frame, text="Your Data, Our Intelligence", font=("Arial", 36, "bold"),
            text_color="#ffffff"
        ).pack(pady=(20, 10))
        ctk.CTkLabel(
            self.frame, text="Upload your CSV or Excel dataset to start analyzing",
            font=("Arial", 16), text_color="#cccccc"
        ).pack(pady=(0, 20))

        self.welcome_label = ctk.CTkLabel(
            self.frame, text="Welcome! Start by uploading your dataset",
            font=("Arial", 18, "bold"), text_color="#ffffff",
            fg_color="#5e7a51", corner_radius=10, width=400, height=50
        )
        self.welcome_label.pack(pady=20)
        self.welcome_label.configure(state="disabled")
        self.fade_in(self.welcome_label, 0, 1, 2)
        self.root.after(5000, lambda: self.fade_out(self.welcome_label, 1, 0, 2))

        upload_button = ctk.CTkButton(
            self.frame, text="Upload Dataset", width=400, height=50, command=self.upload_file,
            fg_color="#5e7a51", text_color="#ffffff", hover_color="#6b9463",
            border_width=2, border_color="#5e7a51", corner_radius=12,
            font=("Arial", 18, "bold")
        )
        upload_button.pack(pady=30)

        ctk.CTkLabel(
            self.frame, text="Supports: .csv, .xlsx", font=("Arial", 14), text_color="#ffffff"
        ).pack()

    def fade_in(self, widget, start_alpha, end_alpha, steps):
        delta = (end_alpha - start_alpha) / steps
        for i in range(steps):
            alpha = start_alpha + delta * i
            widget.after(i * 50, lambda: widget.configure(fg_color=f"#5e7a51", text_color=f"#{int(255*alpha):02x}{int(255*alpha):02x}{int(255*alpha):02x}"))
        widget.after(steps * 50, lambda: widget.configure(fg_color="#5e7a51", text_color="#ffffff"))

    def fade_out(self, widget, start_alpha, end_alpha, steps):
        delta = (end_alpha - start_alpha) / steps
        for i in range(steps):
            alpha = start_alpha + delta * i
            widget.after(i * 50, lambda: widget.configure(fg_color=f"#5e7a51", text_color=f"#{int(255*alpha):02x}{int(255*alpha):02x}{int(255*alpha):02x}"))
        widget.after(steps * 50, lambda: widget.destroy())

    def upload_file(self):
        filetypes = [("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        file_path = filedialog.askopenfilename(title="Select a dataset", filetypes=filetypes)
        if file_path:
            self.main_app.shared_data["dataset_path"] = file_path
            print(f"File selected: {file_path}")
            self.show_success_popup()
            self.main_app.enable_nav_buttons()
        else:
            messagebox.showwarning("No File Selected", "Please select a CSV or Excel file to upload.")

    def show_success_popup(self):
        popup = ctk.CTkToplevel(self.root)
        popup.geometry("300x200")
        popup.title("Success")
        popup.configure(fg_color="#435242")
        popup.resizable(False, False)

        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        popup_x = root_x + (root_width // 2) - (300 // 2)
        popup_y = root_y + (root_height // 2) - (200 // 2)
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(self.root)
        popup.grab_set()
        popup.lift()

        ctk.CTkLabel(
            popup, text="✅", font=("Arial", 36), text_color="#5e7a51"
        ).pack(pady=(20, 5))
        ctk.CTkLabel(
            popup, text="Dataset uploaded successfully!", font=("Arial", 18, "bold"),
            text_color="#ffffff"
        ).pack(pady=5)
        ctk.CTkLabel(
            popup, text="Redirecting to Dashboard...", font=("Arial", 14),
            text_color="#cccccc"
        ).pack(pady=5)

        popup.after(2500, lambda: [popup.destroy(), self.main_app.show_page("Dashboard")])