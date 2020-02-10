import tkinter as tk


class SummaryFrame(tk.Frame):
    ############################################################################################################
    def __init__(self, display):
        tk.Frame.__init__(self)

        self.summary_canvas = tk.Canvas(self,
                                        width=display.summary_canvas_width, height=display.summary_canvas_height,
                                        bg="white", bd=2, highlightthickness=0)
        self.summary_canvas.grid(row=0, column=0)

        self.display = display
        self.summary_canvas_width = display.summary_canvas_width
        self.summary_canvas_height = display.summary_canvas_height

        self.summary_main_title = None

        self.update_summary_display()

    ############################################################################################################
    def clear_summary_display(self):
        self.summary_main_title.destroy()

    ############################################################################################################
    def update_summary_display(self):

        # create the summary display main title
        self.summary_main_title = tk.Label(self.summary_canvas, text="Humans Summary", font="Verdana 16 bold", anchor=tk.W)
        self.summary_main_title.place(x=10, y=10)