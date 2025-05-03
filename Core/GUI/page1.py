
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

class DataUploaderApp:
    def __init__(self, parent):
        self.root = parent
        self.root.configure(fg_color="#435242")
        self.create_widgets()

    def create_widgets(self):
        self.frame = ctk.CTkFrame(master=self.root, width=600, height=300, corner_radius=20, fg_color='#435242', bg_color='#435242')
        self.frame.place(relx=0.5, rely=0.5, anchor="center")
        self.frame.pack_propagate(False)

        self.heading_label1 = ctk.CTkLabel(self.frame, text="Your Data", font=("Arial Bold", 30))
        self.heading_label1.pack(pady=(20, 0))

        self.heading_label2 = ctk.CTkLabel(self.frame, text="Our Intelligence", font=("Arial Bold", 30))
        self.heading_label2.pack(pady=(10,0))

        self.upload_button = ctk.CTkButton(self.frame, text="Upload your DataSet", width=370, height=35, command=self.upload_file,
                                            fg_color="#445344", text_color="white", hover_color="#5e7a51"
                                            , border_width=2 , border_color="#5e7a51" , corner_radius=10)
        self.upload_button.pack(pady=50)

        self.support_label = ctk.CTkLabel(self.frame, text="Supports: .csv, .xlsx", font=("Arial", 14), text_color="white")
        self.support_label.pack()

    def upload_file(self):
        filetypes = [("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        file_path = filedialog.askopenfilename(title="Select a dataset", filetypes=filetypes)
        if file_path:
            print(f"File selected: {file_path}")
            self.show_success_popup()

    def show_success_popup(self):
        popup = ctk.CTkToplevel(self.root)
        popup.geometry("300x200")
        popup.title("Success")
        popup.configure(fg_color="#7FA1C3")
        popup.resizable(False, False)
        popup.grab_set()

        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()

        popup_x = root_x + (root_width // 2) - (300 // 2)
        popup_y = root_y + (root_height // 2) - (200 // 2)
        popup.geometry(f"+{popup_x}+{popup_y}")

        check_label = ctk.CTkLabel(popup, text="✅", font=("Arial", 36), text_color="green", fg_color="#7FA1C3")
        check_label.pack(pady=(20, 5))

        message_label = ctk.CTkLabel(popup, text="Uploaded successfully", font=("Arial Bold", 16), text_color="black", fg_color="#7FA1C3")
        message_label.pack(pady=5)

        close_button = ctk.CTkButton(popup, text="Close", width=100, fg_color="#5A4FFF", hover_color="#4e3dd4", command=popup.destroy)
        close_button.pack(pady=(15, 10))
