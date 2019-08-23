import tkinter as tk


class ButtonFrame(tk.Frame):
    def __init__(self, display):
        super().__init__()

        self.button_canvas = tk.Canvas(self,
                                       width=display.button_canvas_width, height=display.button_canvas_height,
                                       bd=0, highlightthickness=0)
        self.button_canvas.grid(row=0, column=0)

        self.display = display

        self.button_height = 2
        self.button_width = 8

        self.next_button = None
        self.run_button = None
        self.save_button = None
        self.quit_button = None

        self.create_buttons()

    ############################################################################################################
    def create_buttons(self):

        self.next_button = tk.Button(self.button_canvas, text="Next", fg="black",
                                     height=self.button_height, width=self.button_width,
                                     command=self.display.update)
        self.next_button.grid(row=0, column=0, sticky=tk.W)

        self.run_button = tk.Button(self.button_canvas, text="Run", fg="black",
                                    height=self.button_height, width=self.button_width,
                                    command=self.display.run_simulation)
        self.run_button.grid(row=0, column=1, sticky=tk.W)

        self.save_button = tk.Button(self.button_canvas, text="Save", fg="black",
                                     height=self.button_height, width=self.button_width,
                                     command=self.display.save_simulation)
        self.save_button.grid(row=0, column=2, sticky=tk.W)

        self.quit_button = tk.Button(self.button_canvas, text="Quit", fg="black",
                                     height=self.button_height, width=self.button_width,
                                     command=self.display.quit_simulation)
        self.quit_button.grid(row=0, column=3, sticky=tk.E)
