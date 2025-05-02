import customtkinter as ctk
import tkinter as tk

#6482AD
#7FA1C3
#E2DAD6
#F5EDED
class DataDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Visualization Dashboard")
        self.root.geometry("1000x600")
        self.root.configure(fg_color="#F5EDED")
        

        # ====== Customization Options ======
        self.button_width = 180
        self.button_height = 40
        self.button_font = ("Arial", 14)
        self.button_pad_y = 10
        self.button_pad_x = 15
        self.button_color = "#6482AD"
        self.button_text_color = 'white'
        self.button_hover_color = "#7FA1C3"

        self.tab_names = [
            "Dataset summary, Structure", "Ydata porfiling", "Show hist",
            "Show heatmap", "Show pairplot", "Show Barchart", "Generate Your own plot"
        ]

        self.frames = {}

        self.setup_layout()

    def setup_layout(self):
        """Initialize sidebar and content area."""
        self.create_sidebar()
        self.create_content_area()
        self.show_frame(self.tab_names[0])

    def create_sidebar(self):
        """Sidebar with navigation buttons."""
        self.sidebar = ctk.CTkFrame(self.root, width=200, fg_color='#E2DAD6')
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
            command=self.go_to_next_page  # Define this method to handle navigation
        )
        next_page_button.pack(side="bottom", pady=10, padx=15)    

    def create_content_area(self):
        """Main area to display tab contents."""
        self.content_area = ctk.CTkFrame(self.root, fg_color='#E2DAD6',corner_radius=10)
        self.content_area.pack(side="right", expand=True, fill="both",pady=10, padx=10)

        for tab_name in self.tab_names:
            frame = ctk.CTkFrame(self.content_area,fg_color='#E2DAD6', corner_radius=10)
            label = ctk.CTkLabel(frame, text=f"Content for: {tab_name}", font=("Arial", 20), text_color="#6482AD")
            label.pack(pady=20)
            self.frames[tab_name] = frame

    def show_frame(self, name):
        """Switch to the selected tab."""
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[name].pack(expand=True, fill="both",padx=10, pady=10)


    def go_to_next_page(self):
        """Handle navigation to the next page."""
        # Here you can implement the logic to navigate to the next page
        print("Navigating to the next page...")
        # For example, you could create a new window or frame for the next page

if __name__ == "__main__":
    root = ctk.CTk()
    app = DataDashboardApp(root)
    root.mainloop()