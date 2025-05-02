import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

#6482AD
#7FA1C3
#E2DAD6
#F5EDED

class DataUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Uploader")
        self.root.geometry("1000x600")
        self.root.config(background="#E2DAD6")
        
        

        self.create_widgets()

    def create_widgets(self):
        # Main Frame
        self.frame = ctk.CTkFrame(master=self.root, width=600, height=300, corner_radius=20, fg_color='#6482AD',bg_color='#E2DAD6')
        self.frame.place(relx=0.5, rely=0.5, anchor="center")  
        self.frame.pack_propagate(False)
        
        #Heading
        self.heading_label1 = ctk.CTkLabel(self.frame, text="Your Data,", font=("Arial Bold", 24))
        self.heading_label1.pack(pady=(20, 0))

        self.heading_label2 = ctk.CTkLabel(self.frame, text="Our Intelligence", font=("Arial Bold", 24), text_color="black")
        self.heading_label2.pack(pady=(10,0))

        #Upload Button
        self.upload_button = ctk.CTkButton(self.frame, text="Upload your DataSet", width=370, height=35, command=self.upload_file,fg_color='#7FA1C3',hover_color='#E2DAD6',text_color='white')
        self.upload_button.pack(pady=50)

        #File format info
        self.support_label = ctk.CTkLabel(self.frame, text="Supports: .csv, .xlsx", font=("Arial", 14), text_color="white")
        self.support_label.pack()

    def upload_file(self):
        filetypes = [("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        file_path = filedialog.askopenfilename(title="Select a dataset", filetypes=filetypes)
        if file_path:
            print(f"File selected: {file_path}")  # You can replace this with actual processing

            self.show_success_popup()

    def show_success_popup(self):
        popup = ctk.CTkToplevel(self.root)
        popup.geometry("300x200")
        popup.title("Success")
        popup.configure(fg_color="#7FA1C3")
        popup.resizable(False, False)
        popup.grab_set()  # Modal behavior


           # Position the popup directly above the root window
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()

        popup_x = root_x + (root_width // 2) - (300 // 2)  # Center horizontally
        popup_y = root_y + (root_height // 2) - (200 // 2)  # Center vertically
        popup.geometry(f"+{popup_x}+{popup_y}")

        # Green checkmark
        check_label = ctk.CTkLabel(popup, text="✅", font=("Arial", 36), text_color="green", fg_color="#7FA1C3")
        check_label.pack(pady=(20, 5))

        # Message
        message_label = ctk.CTkLabel(popup, text="Uploaded successfully", font=("Arial Bold", 16),
                                     text_color="black", fg_color="#7FA1C3")
        message_label.pack(pady=5)

        # Close Button
        close_button = ctk.CTkButton(popup, text="Close", width=100, fg_color="#5A4FFF",
                                     hover_color="#4e3dd4", command=popup.destroy)
        close_button.pack(pady=(15, 10))        

if __name__ == "__main__":
    root = ctk.CTk()
    app = DataUploaderApp(root)
    root.mainloop()
    