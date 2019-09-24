import tkinter as tk
from src import config


class MainFrame(tk.Frame):
    ############################################################################################################
    def __init__(self, display, the_world, image_dict):
        super().__init__()

        self.display = display
        self.the_world = the_world
        self.image_dict = image_dict

        self.main_canvas = tk.Canvas(self,
                                     height=display.main_canvas_height, width=display.main_canvas_width,
                                     bd=0, highlightthickness=0, bg='#000000',
                                     scrollregion=(0, 0, display.main_canvas_width, display.main_canvas_height))

        self.game_frame = tk.Frame(self.main_canvas)
        self.game_frame.grid(row=0, column=0)

        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        self.hsb = tk.Scrollbar(self, orient="horizontal", command=self.main_canvas.xview)
        self.main_canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.vsb.grid(sticky=tk.N+tk.S+tk.E)
        self.hsb.grid(sticky=tk.E+tk.W+tk.S)
        self.main_canvas.grid(row=0, column=0)
        self.main_canvas.create_window((4, 4), window=self.game_frame, anchor="nw")

        self.game_frame.bind("<Configure>", lambda event, canvas=self.main_canvas: self.on_frame_configure(canvas))

        self.main_canvas.bind('<Double-Button-1>', self.main_canvas_on_double_click)

    ############################################################################################################
    @staticmethod
    def on_frame_configure(canvas):
        canvas.configure(scrollregion=canvas.bbox("all"))

    ############################################################################################################
    def main_canvas_on_double_click(self, event):
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        print("Click Event:", x, y)

    ############################################################################################################
    def draw_terrain(self):
        max_size = config.World.num_tiles

        for i in range(max_size):
            for j in range(max_size):

                if i == 0 or i == max_size-1 or j == 0 or j == max_size-1:
                    the_image = self.display.image_dict['water']
                else:
                    the_image = self.display.image_dict['terrain']

                self.main_canvas.create_image(i*config.World.tile_size, j*config.World.tile_size, anchor=tk.NW, image=the_image)

    ############################################################################################################
    def draw_objects(self):
        pass

    ############################################################################################################
    def draw_humans(self):
        for i in range(len(self.the_world.human_list)):
            human = self.the_world.human_list[i]
            self.main_canvas.create_image(human.x, human.y, anchor=tk.NW, image=self.display.image_dict['human'])

    ############################################################################################################
    def draw_animals(self):
        pass


