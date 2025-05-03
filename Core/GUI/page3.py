import customtkinter as ctk

class DataPreprocessingApp:
        def __init__(self, parent):
            self.root = parent
            self.create_widgets()

        def create_widgets(self):
            self.container = ctk.CTkFrame(self.root, fg_color="#435242")
            self.container.pack(expand=True, fill="both", padx=10, pady=10)

            ctk.CTkLabel(
                self.container,
                text="Some info, alerts from ydata portfolio that would help us",
                font=("Arial", 20, "bold"),
                text_color="white"
            ).pack(pady=(10, 20))

            cards_container = ctk.CTkFrame(self.container, fg_color="#435242")
            cards_container.pack(expand=True, fill="both", padx=20, pady=10)

            cards_container.grid_rowconfigure((0, 1), weight=1)
            cards_container.grid_columnconfigure((0, 1), weight=1)

            cards_data = [
                ("missing values methods:", ["drop", "fill", "simple imputer"]),
                ("categorical columns Encoding:", ["One-Hot", "Label Encoding"]),
                ("data before Normalization:", ["Min-Max", "Z-score"]),
                ("Handle Outliers:", ["Remove", "Cap", "Transform", "Ignore"]),
            ]

            for idx, (label, options) in enumerate(cards_data):
                row, col = divmod(idx, 2)
                card = self.create_card(cards_container, label, options)
                card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        def create_card(self, parent, text, options):
            frame = ctk.CTkFrame(parent, fg_color="#445344", corner_radius=15,
                                border_width=2, border_color="#5e7a51")
            ctk.CTkLabel(frame, text=text, font=("Arial", 16),
                        text_color="white", wraplength=220).pack(pady=(20, 10))
            ctk.CTkOptionMenu(frame, values=options,
                            fg_color="white", text_color="black",
                            button_color="#445344", button_hover_color="#5e7a51").pack(pady=(0, 20))
            frame.bind("<Enter>", lambda e: frame.configure(fg_color="#5e7a51"))
            frame.bind("<Leave>", lambda e: frame.configure(fg_color="#445344"))
            return frame
