import tkinter as tk
from PIL import Image, ImageTk
import sys
from src.display import button_frame, summary_frame, main_frame
from src import config


class Display:
    ############################################################################################################
    def __init__(self, the_world):
        self.the_world = the_world
        self.turn = 0
        self.image_dict = None
        self.running = False
        self.timing_freq = 0

        self.world_height = None
        self.world_width = None

        self.root = None
        self.root_height = 800
        self.root_width = 1600

        self.main_frame = None
        self.main_canvas = None
        self.main_canvas_height = self.root_height - 50
        self.main_canvas_width = self.root_width - 500 - 15

        self.summary_frame = None
        self.summary_canvas_height = self.root_height - 35
        self.summary_canvas_width = 500

        self.button_frame = None
        self.button_canvas_height = 20
        self.button_canvas_width = self.root_width

        self.create_main_window()
        self.create_main_frame()
        self.create_summary_frame()

        self.create_buttons()

        self.load_images()
        self.main_frame.draw_terrain()
        self.main_frame.draw_objects()
        self.main_frame.draw_humans()
        self.main_frame.draw_animals()

    ############################################################################################################
    def create_main_window(self):
        self.root = tk.Tk()
        self.root.resizable(0, 0)
        self.root.title("Humans: Time {}".format(self.turn))

    ############################################################################################################
    def create_main_frame(self):
        self.main_frame = main_frame.MainFrame(self, self.the_world, self.image_dict)
        self.main_frame.grid(row=0, column=0)

    ############################################################################################################
    def create_summary_frame(self):
        self.summary_frame = summary_frame.SummaryFrame(self)
        self.summary_frame.grid(row=0, column=1, padx=0, pady=0, ipadx=0, ipady=0)

    ############################################################################################################
    def create_buttons(self):
        self.button_frame = button_frame.ButtonFrame(self)
        self.button_frame.grid(row=1, column=0, columnspan=2, padx=0, pady=0, ipadx=0, ipady=0)

    ############################################################################################################
    def load_images(self):
        self.image_dict = {}

        image = Image.open('assets/images/terrain.png')
        terrain_image = ImageTk.PhotoImage(image.resize((config.World.tile_size, config.World.tile_size)))
        self.image_dict['terrain'] = terrain_image

        image = Image.open('assets/images/water.gif')
        water_image = ImageTk.PhotoImage(image.resize((config.World.tile_size, config.World.tile_size)))
        self.image_dict['water'] = water_image

        image = Image.open('assets/images/hunter.png')
        human_image = ImageTk.PhotoImage(image.resize((config.World.tile_size, config.World.tile_size)))
        self.image_dict['human'] = human_image

    ############################################################################################################
    def run_simulation(self):
        if self.running:
            self.running = False
            self.button_frame.run_button.config(text="Run")
        else:
            self.running = True
            self.button_frame.run_button.config(text="Pause")

        while self.running:
            self.update()

    ############################################################################################################
    def update(self):

        self.turn += 1
        self.root.title("Humans: Time {}".format(self.turn))

        self.the_world.next_turn()
        self.main_frame.main_canvas.delete("all")
        self.main_frame.draw_terrain()
        self.main_frame.draw_objects()
        self.main_frame.draw_animals()
        self.main_frame.draw_humans()

        self.summary_frame.clear_summary_display()
        self.summary_frame.update_summary_display()

        self.root.update()

    ############################################################################################################
    def save_simulation(self):
        pass

    ############################################################################################################
    def quit_simulation(self):
        if self.running:
            self.run_simulation()
        sys.exit(1)
