import os
import csv
import time
import tkinter as tk
import tkinter.messagebox
import tkinter.simpledialog
import tkinter.ttk as ttk
from tkinter import *

from .arguments import get_args
from .env.Env1D import Env1DStatic, Env1DDynamic
from .env.Env2D import Env2DStatic, Env2DDynamic


class HumanPlay:
    def __init__(self):

        self.user = tk.simpledialog.askstring('Enter user', 'Enter your first name (all-lowercase):', parent=Tk())
        self.session_id = str(time.time())[-6:]

        # Env
        self.ENV_DICT = {
            1: '1D Static (Plan 1)',
            2: '1D Static (Plan 2)',
            3: '1D Static (Plan 3)',
            4: '1D Dynamic',
            5: '2D Static (Dense Plan)',
            6: '2D Static (Sparse Plan)',
            7: '2D Dynamic (Dense Plan)',
            8: '2D Dynamic (Sparse Plan)'
        }
        self.env = None
        self.env_choose = 1  # goes from 1-8 (only 1D and 2D plans supported)
        self.args = get_args()
        self.environment_width = None
        self.environment_height = None
        self.window_size = None
        self.max_steps = None
        self.max_bricks = None

        self.brick_action_idx = None
        self.L_action_idx = None
        self.R_action_idx = None
        self.U_action_idx = None
        self.D_action_idx = None

        # State
        self.step_count = 0
        self.brick_count = 0
        self.reward = 0
        self.episode_reward = 0
        self.environment_memory = None
        self.observation = None
        self.done = None

        # GUI
        self.menu_window = Tk()
        self.menu_window.title('Choose Plan')
        self.menu_window.geometry('+200+0')
        self.game_window = None
        self.game_mode = None

        self.plan_window = None
        self.plan_canvas = None

        self.canvas = None
        self.canvas_width = None
        self.canvas_height = None
        self.num_steps_id = None
        self.num_bricks_id = None
        self.num_reward_id = None
        self.num_totalreward_id = None

        self.build_menu_window()
        self.reset_game(self.env_choose)

    def reset_game(self, env_choose):
        self.env_choose = env_choose
        self.step_count = 0
        self.brick_count = 0
        self.reward = 0
        self.episode_reward = 0
        self.done = False
        self.done = False

        if self.env is not None:
            self.env.close()

        if env_choose <= 4:
            self.display_state = [0 for i in range(5)]
            self.canvas_width = 400
            self.canvas_height = 600

            self.L_action_idx = 0
            self.R_action_idx = 1
            self.brick_action_idx = 2

            if env_choose == 1:
                self.env = Env1DStatic(self.args)
                self.env.set_plan_choose(0)

            elif env_choose == 2:
                self.env = Env1DStatic(self.args)
                self.env.set_plan_choose(1)

            elif env_choose == 3:
                self.env = Env1DStatic(self.args)
                self.env.set_plan_choose(2)

            elif env_choose == 4:
                self.env = Env1DDynamic(self.args)

        else:
            self.display_state = [0 for i in range(49)]
            self.canvas_width = 400
            self.canvas_height = 400

            self.L_action_idx = 0
            self.R_action_idx = 1
            self.D_action_idx = 2
            self.U_action_idx = 3
            self.brick_action_idx = 4

            if env_choose == 5:
                self.env = Env2DStatic(self.args)
                self.env.set_plan_choose(0)

            elif env_choose == 6:
                self.env = Env2DStatic(self.args)
                self.env.set_plan_choose(1)

            elif env_choose == 7:
                self.env = Env2DDynamic(self.args)
                self.env.set_plan_choose(1)

            elif env_choose == 8:
                self.env = Env2DDynamic(self.args)
                self.env.set_plan_choose(0)

        self.observation = self.env.reset()
        self.reset_game_window()
        self.reset_plan_window()
        self.update_canvas()

    def build_menu_window(self):
        bottom = tk.Frame(self.menu_window).pack(side="bottom")
        right = tk.Frame(self.menu_window).pack(side="right")

        self.game_mode = tk.StringVar()
        self.game_mode_combobox = ttk.Combobox(width=18,
                                               textvariable=self.game_mode,
                                               state="readonly",
                                               justify='center',
                                               values=['Training Mode', 'Evaluation Mode'])
        self.game_mode_combobox.pack()
        self.game_mode_combobox.current(0)
        self.game_mode_combobox.bind('<<ComboboxSelected>>', self.set_mode)

        self.button_env1 = tk.Button(bottom, text="1D Static (Plan 1)", fg="orange",
                                     command=lambda x=1: self.reset_game(x)).pack()
        self.button_env2 = tk.Button(bottom, text="1D Static (Plan 2)", fg="orange",
                                     command=lambda x=2: self.reset_game(x)).pack()
        self.button_env3 = tk.Button(bottom, text="1D Static (Plan 3)", fg="orange",
                                     command=lambda x=3: self.reset_game(x)).pack()
        self.button_env4 = tk.Button(bottom, text="1D Dynamic", fg="green",
                                     command=lambda x=4: self.reset_game(x)).pack()
        self.button_env5 = tk.Button(bottom, text="2D Static (Dense Plan)", fg="purple",
                                     command=lambda x=5: self.reset_game(x)).pack()
        self.button_env6 = tk.Button(bottom, text="2D Static (Sparse Plan)", fg="purple",
                                     command=lambda x=6: self.reset_game(x)).pack()
        self.button_env7 = tk.Button(bottom, text="2D Dynamic (Dense Plan)", fg="red",
                                     command=lambda x=7: self.reset_game(x)).pack()
        self.button_env8 = tk.Button(bottom, text="2D Dynamic (Sparse Plan)", fg="red",
                                     command=lambda x=8: self.reset_game(x)).pack()
        self.button_env9 = tk.Button(right, text="End Episode", fg="black",
                                     command=self.upon_episode_completion).pack()

    def reset_plan_window(self):
        if self.plan_window is not None:
            self.plan_window.destroy()
            self.plan_window = None

        if self.env_choose in [4, 7, 8]:
            self.plan_window = Tk()
            self.plan_window.geometry('+700+0')
            self.plan_window.title('Dynamic Plan')

            self.plan_canvas = Canvas(
                self.plan_window,
                width=500,
                height=500)

            if self.env_choose == 4:
                self.plan_canvas.create_rectangle(25, 25, 475, 475, outline='gray75', fill='white')
                for i in range(49):
                    self.plan_canvas.create_line(25, 25 + 9 * (i + 1), 475, 25 + 9 * (i + 1), fill='gray85')
                for i in range(29):
                    self.plan_canvas.create_line(25 + 15 * (i + 1), 25, 25 + 15 * (i + 1), 475, fill='gray85')
                for i in range(6):
                    self.plan_canvas.create_text(15, 475 - i * 90, text=str(i * 10), fill='gray60', font=('Arial', 10))
                for i in range(4):
                    self.plan_canvas.create_text(25 + i * 150, 485, text=str(i * 10), fill='gray60', font=('Arial', 10))
                for i, value in enumerate(self.env.plan):
                    self.plan_canvas.create_rectangle(25 + i * 15,
                                                      475,
                                                      25 + (i + 1) * 15,
                                                      475 - 9 * value,
                                                      outline='SeaGreen2',
                                                      fill='SeaGreen2')
            else:
                self.plan_canvas.create_rectangle(30, 30, 470, 470, outline='gray75', fill='white')
                for i in range(20):
                    self.plan_canvas.create_line(30, 30 + 22 * (i + 1), 470, 30 + 22 * (i + 1), fill='gray85')
                    self.plan_canvas.create_line(30 + 22 * (i + 1), 30, 30 + 22 * (i + 1), 470, fill='gray85')
                for i in range(5):
                    self.plan_canvas.create_text(30 + i * 110, 480, text=str(i * 5), fill='gray60', font=('Arial', 10))
                    self.plan_canvas.create_text(20, 470 - i * 110, text=str(i * 5), fill='gray60', font=('Arial', 10))
                for i in range(3, 23):
                    for j in range(3, 23):
                        if self.env.plan[i][j] == 1:
                            self.plan_canvas.create_rectangle(30 + 22 * (j - 3),
                                                              30 + 22 * (i - 3),
                                                              30 + 22 * (j - 2),
                                                              30 +  + 22 * (i - 2),
                                                              outline='SeaGreen2',
                                                              fill='SeaGreen2')
            self.plan_canvas.pack()

    def reset_game_window(self):
        if self.game_window is not None:
            self.game_window.destroy()

        self.game_window = Tk()
        self.game_window.geometry('+250+0')
        self.game_window.title('C.A.M. Human Performance Benchmarking')
        self.game_window.bind("<Key>", self.keypress)
        self.game_window.bind("<Left>", lambda event: self.move('L'))
        self.game_window.bind("<Right>", lambda event: self.move('R'))
        self.game_window.bind("<Up>", lambda event: self.move('U'))
        self.game_window.bind("<Down>", lambda event: self.move('D'))

        self.canvas = Canvas(
            self.game_window,
            width=self.canvas_width,
            height=self.canvas_height)

        if self.env_choose <= 4:
            self.canvas.create_rectangle(162, 35, 237, 485, outline='gray75')
            for i in range(49):
                self.canvas.create_line(162, 35 + 9 * (i + 1), 237,  35 + 9 * (i + 1), fill='gray85')
            for i in range(4):
                self.canvas.create_line(162 + 15 * (i + 1), 35, 162 + 15 * (i + 1), 485, fill='gray85')
            self.canvas.create_text(152, 485, text=str(0), fill='gray60', font=('Arial', 10))
            self.canvas.create_text(152, 260, text=str(25), fill='gray60', font=('Arial', 10))
            self.canvas.create_text(152, 35, text=str(50), fill='gray60', font=('Arial', 10))
            self.canvas.create_text(200, 497, text='*', fill='red3', font=('Arial', 18))

            self.txt_plan_id = self.canvas.create_text(200, 590, text='plan: ' + self.ENV_DICT[self.env_choose],
                                                       fill='gray60', font=('Arial', 10))

            if self.in_training_mode():
                self.txt_steps_id = self.canvas.create_text(50, 530, text='steps taken',
                                                            fill='gray60', font=('Arial', 12))
                self.txt_bricks_id = self.canvas.create_text(150, 530, text='bricks used',
                                                             fill='gray60', font=('Arial', 12))
                self.num_steps_id = self.canvas.create_text(50, 555, text=str(0),
                                                            fill='gray40', font=('Arial', 18))
                self.num_bricks_id = self.canvas.create_text(150, 555, text=str(0),
                                                             fill='gray40', font=('Arial', 18))
                self.txt_reward_id = self.canvas.create_text(250, 530, text='step reward',
                                                             fill='gray60', font=('Arial', 12))
                self.txt_totalreward_id = self.canvas.create_text(350, 530, text='total reward',
                                                                  fill='gray60', font=('Arial', 12))
                self.num_reward_id = self.canvas.create_text(250, 555, text=str(0),
                                                           fill='gray40', font=('Arial', 18))
                self.num_totalreward_id = self.canvas.create_text(350, 555, text=str(0),
                                                           fill='gray40', font=('Arial', 18))
            else:
                self.txt_steps_id = self.canvas.create_text(150, 530, text='steps taken',
                                                            fill='gray60', font=('Arial', 12))
                self.txt_bricks_id = self.canvas.create_text(250, 530, text='bricks used',
                                                             fill='gray60', font=('Arial', 12))
                self.num_steps_id = self.canvas.create_text(150, 555, text=str(0),
                                                            fill='gray40', font=('Arial', 18))
                self.num_bricks_id = self.canvas.create_text(250, 555, text=str(0),
                                                             fill='gray40', font=('Arial', 18))
        else:
            for i in range(7):
                for j in range(7):
                    self.display_state[7 * i + j] = self.canvas.create_rectangle(60 + 40 * j,
                                                                                 25 + 40 * i,
                                                                                 60 + 40 * (j + 1),
                                                                                 25 + 40 * (i + 1),
                                                                                 fill='white')
            self.canvas.create_rectangle(60, 25, 340, 305, outline='gray75')
            self.canvas.create_line(60, 65, 340, 65, fill='gray85')
            self.canvas.create_line(60, 105, 340, 105, fill='gray85')
            self.canvas.create_line(60, 145, 340, 145, fill='gray85')
            self.canvas.create_line(60, 185, 340, 185, fill='gray85')
            self.canvas.create_line(60, 225, 340, 225, fill='gray85')
            self.canvas.create_line(60, 265, 340, 265, fill='gray85')
            self.canvas.create_line(100, 25, 100, 305, fill='gray85')
            self.canvas.create_line(140, 25, 140, 305, fill='gray85')
            self.canvas.create_line(180, 25, 180, 305, fill='gray85')
            self.canvas.create_line(220, 25, 220, 305, fill='gray85')
            self.canvas.create_line(260, 25, 260, 305, fill='gray85')
            self.canvas.create_line(300, 25, 300, 305, fill='gray85')
            self.txt_plan_id = self.canvas.create_text(200, 390, text='plan: ' + self.ENV_DICT[self.env_choose],
                                                       fill='gray60', font=('Arial', 10))

            if self.in_training_mode():
                self.txt_steps_id = self.canvas.create_text(50, 330, text='steps taken',
                                                            fill='gray60', font=('Arial', 12))
                self.txt_bricks_id = self.canvas.create_text(150, 330, text='bricks used',
                                                             fill='gray60', font=('Arial', 12))
                self.num_steps_id = self.canvas.create_text(50, 355, text=str(0),
                                                            fill='gray40', font=('Arial', 18))
                self.num_bricks_id = self.canvas.create_text(150, 355, text=str(0),
                                                             fill='gray40', font=('Arial', 18))
                self.txt_reward_id = self.canvas.create_text(250, 330, text='step reward',
                                                             fill='gray60', font=('Arial', 12))
                self.num_reward_id = self.canvas.create_text(250, 355, text=str(0),
                                                             fill='gray40', font=('Arial', 18))
                self.txt_totalreward_id = self.canvas.create_text(350, 330, text='total reward',
                                                                  fill='gray60', font=('Arial', 12))
                self.num_totalreward_id = self.canvas.create_text(350, 355, text=str(0),
                                                                  fill='gray40', font=('Arial', 18))
            else:
                self.txt_steps_id = self.canvas.create_text(150, 330, text='steps taken',
                                                            fill='gray60', font=('Arial', 12))
                self.txt_bricks_id = self.canvas.create_text(250, 330, text='bricks used',
                                                             fill='gray60', font=('Arial', 12))
                self.num_steps_id = self.canvas.create_text(150, 355, text=str(0),
                                                            fill='gray40', font=('Arial', 18))
                self.num_bricks_id = self.canvas.create_text(250, 355, text=str(0),
                                                             fill='gray40', font=('Arial', 18))

        self.canvas.pack()

    def update_canvas(self):
        observation = self.observation.squeeze()
        if self.env_choose <= 4:
            for i in range(5):
                self.canvas.delete(self.display_state[i])
                value = observation[i]
                if value == -1:
                    self.display_state[i] = self.canvas.create_rectangle(162 + 15 * i,
                                                                         35,
                                                                         162 + 15 * (i + 1),
                                                                         485,
                                                                         fill='grey30', outline='gray30')
                elif value > 0:
                    self.display_state[i] = self.canvas.create_rectangle(162 + 15 * i,
                                                                         485,
                                                                         162 + 15 * (i + 1),
                                                                         485 - int(value) * 9,
                                                                         fill='coral2', outline='coral2')
        else:
            for i in range(7):      # row
                for j in range(7):  # column
                    self.canvas.delete(self.display_state[7 * i + j])
                    value = observation[7 * i + j]
                    if value == -1:
                        self.display_state[7 * i + j] = self.canvas.create_rectangle(60 + 40 * j,
                                                                                 25 + 40 * i,
                                                                                 60 + 40 * (j + 1),
                                                                                 25 + 40 * (i + 1),
                                                                                 fill='grey30', outline='grey30')
                    elif value == 1:
                        self.display_state[7 * i + j] = self.canvas.create_rectangle(60 + 40 * j,
                                                                                 25 + 40 * i,
                                                                                 60 + 40 * (j + 1),
                                                                                 25 + 40 * (i + 1),
                                                                                 fill='coral2', outline='coral2')
            self.canvas.create_text(201, 169, text='*', fill='red3', font=('Arial', 20), anchor=tk.CENTER)

        self.canvas.itemconfigure(self.num_steps_id, text=str(self.step_count))
        self.canvas.itemconfigure(self.num_bricks_id, text=str(self.brick_count))

        if self.in_training_mode():
            self.canvas.itemconfigure(self.num_reward_id, text=str(self.reward))
            self.canvas.itemconfigure(self.num_totalreward_id, text=str(self.episode_reward))

    def upon_episode_completion(self):
        save = tkinter.messagebox.askquestion("Save", "Episode ended - save the result?")

        if save == 'yes':
            path = 'results/human_results' + '_' + self.user + '_' + self.session_id + '.csv'
            self.log_result(path)
            tkinter.messagebox.showinfo("Saved.", "Episode results appended to:\n" + os.path.abspath(path))
        else:
            tkinter.messagebox.showinfo("Not saved.", "Episode results discarded.")

        self.reset_game(self.env_choose)

    def log_result(self, path):
        if not os.path.exists('results'):
            os.makedirs('results')

        with open(path, 'a', newline='') as log_file:
            schema = ['user', 'env', 'game_mode', 'iou', 'reward', 'num_steps', 'num_bricks']
            writer = csv.DictWriter(log_file, fieldnames=schema)
            writer.writerow({'user': self.user,
                             'env': self.ENV_DICT[self.env_choose],
                             'game_mode': self.game_mode.get(),
                             'iou': self.env._iou(),
                             'reward': self.episode_reward,
                             'num_steps': self.step_count,
                             'num_bricks': self.brick_count})

    def move(self, direction):
        if direction == 'L':
            self.observation, self.reward, self.done = self.env.step(self.L_action_idx)
            self.step_count += 1
            self.after_move()
        elif direction == 'R':
            self.observation, self.reward, self.done = self.env.step(self.R_action_idx)
            self.step_count += 1
            self.after_move()
        elif direction == 'D' and self.env_choose > 4:
            self.observation, self.reward, self.done = self.env.step(self.D_action_idx)
            self.step_count += 1
            self.after_move()
        elif direction == 'U' and self.env_choose > 4:
            self.observation, self.reward, self.done = self.env.step(self.U_action_idx)
            self.step_count += 1
            self.after_move()

    def after_move(self):
        self.update_canvas()
        if self.done:
            self.upon_episode_completion()

    def keypress(self, event):
        if event.char == ' ':  # Space bar (drop brick in current location)
            self.observation, self.reward, self.done = self.env.step(self.brick_action_idx)
            self.episode_reward += self.reward
            self.step_count += 1
            self.brick_count += 1
            self.after_move()

        elif event.char == '\x1b':  # Escape key (stop playing)
            self.upon_episode_completion()
            self.game_window.destroy()
            self.game_window = None

    def in_training_mode(self):
        return self.game_mode.get() == "Training Mode"

    def set_mode(self, event):
        if self.in_training_mode():
            self.reset_game(self.env_choose)
        else:
            self.reset_game(self.env_choose)

    def mainloop(self):
        self.menu_window.mainloop()
 