#########################################################
#
#   DriveSimulator.py
#   © 2022 Yeonjun Kim (김연준) <kyjunclsrm@gmail.com>
#
#########################################################

from asynchat import simple_producer
from cmath import pi
import collections
import numpy as np
import pygame
import random
import os
import time
import math

from Drive_AI import *

class Agent(object):
    def __init__(self, initX, initY):
        # Agent properties
        self.W = 40
        self.H = 60
        self.V = 15 # Velocity
        
        self.x = initX - self.W/2
        self.y = initY
        self.rotation = 0.0
        self.total_reward = 0.0 # Accumulated reward
        self.step_reward = 0.0 # Reward for each step
        self.win_count = 0
        self.win_data = [0]
        
        self.original_image = pygame.image.load("carimg.png").convert()
        self.image = self.original_image
        self.rect = pygame.Rect(self.x, self.y, self.W, self.H)

        self.episode_count = 0
        self.brain = Brain()
    
    def reset_agent(self, initX, initY):
        self.x = initX - self.W/2
        self.y = initY
        self.rotation = 0.0
        self.total_reward = 0.0 # Accumulated reward
        self.step_reward = 0.0 # Reward for each step
    
    def update(self):
        self.x -= self.V * math.sin(self.rotation)
        self.y -= self.V * math.cos(self.rotation)
        self.image = pygame.transform.rotate(self.original_image, self.rotation*180/math.pi)
        #self.rect = pygame.Rect(self.x-(self.W/2), self.y-(self.H/2), self.W, self.H)
        
    def decide_action(self, s_t):
        if self.episode_count < NUM_EPOCHS_OBSERVE:
            return np.random.randint(0,3), 0 # Do random action (In early episodes)
        else:
            return self.brain.think(s_t) #Return action with the largest Q

    def process_step(self, s_t, a_t, r_t, s_tp1, sim_over):
        # Append data to Experience Replay Memory
        self.brain.exp_memory.append((s_t, a_t, r_t, s_tp1, sim_over))
        #if r_t == 1.0 or r_t == -1.0:
        #    self.brain.imp_exp_memory.append((s_t, a_t, r_t, s_tp1, sim_over))


    def train(self):
        #Visualize Accuracy (Every 100 Episodes)
        if self.episode_count % 100 == 0:
            self.win_data.append(self.win_count)
            print('->', self.episode_count,'Episode째, 승률:', self.win_count,'/100 Episode')
            self.brain.model.save("Model_20220518")
            self.visualize_result(self.episode_count, self.win_data)
            self.win_count = 0

        #Train
        if self.episode_count >= NUM_EPOCHS_OBSERVE:

            X, Y = self.brain.get_next_batch()

            ####DEBUG####
            #print(X)
            #print(np.shape(X))
                
            loss = self.brain.model.train_on_batch(X,Y)
            if (self.episode_count + 1) % 100 == 0:
                print("Epoch {:04d}/{:d} | Loss : {:.5f} | Win Count {:d}".format(self.episode_count+1, NUM_EPOCHS, loss, self.win_count))
            #fout.write("{:04d}t{:.5f}t{:d}n".format(e+1, loss, num_wins))
                
            if self.brain.epsilon > FINAL_EPSILON:
                self.brain.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS
        
    def visualize_result(self, edata, ydata):
        xdata = range(0, edata+1, 100)
        
        plt.xlabel('Episodes')
        plt.ylabel('Wins (Recent 100 Episodes)')
        plt.plot(xdata, ydata)
        plt.show()
        
        

class DriveSimulator(object):
    def __init__(self):
        # Initialize pygame
        os.environ["DSL_VIDEODRIVER"] = "dummy"
        pygame.init()

        # Initialize Constants . . .
        # Colors
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_AGENT = (102,255,255)
        self.COLOR_PATH = (255,255,102)
        self.COLOR_OBS = (204,0,0)

        # Screen size
        self.SCREEN_W = 500
        self.SCREEN_H = 800
        self.STATUS_H = 100 # Agent status monitor below screen
        self.PASS_H = 100
        
        # Obs(Obstacle) properties
        self.obs_r = 0 # Radius (Randomly assigned each game)
        #self.OBS_Vv = 20 # Vertical velocity

        # Others
        self.episode_count = 0
        self.PATH_XPOS = self.SCREEN_W // 2
        self.FONT_SIZE = 16
        self.CUSTOM_EVENT = pygame.USEREVENT + 1
        self.font = pygame.font.SysFont("nanumgothicbold", self.FONT_SIZE)

        
        # Initialize screen
        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H + self.STATUS_H))
        self.clock = pygame.time.Clock()

        # Initialize Agent
        self.agent = Agent(self.PATH_XPOS, 750)


    def reset(self):
        self.episode_count += 1
        self.agent.episode_count += 1

        # Initialize variables
        self.sim_state = np.array([])
        self.sim_prev_state = np.array([])
        self.sim_ticks = 0
        self.sim_over = False
        self.sim_over_why = ''

        self.obs_r = 100#random.randint(75,125)
        self.obs_x = random.randint(self.obs_r, self.SCREEN_W - self.obs_r)
        self.obs_y = 350

        # Initialize screen
        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H + self.STATUS_H))
        self.clock = pygame.time.Clock()
        
        self.agent.reset_agent(self.PATH_XPOS, 750)

        # Initialize sim_state
        self.sim_state = self.get_sim_state()

    def get_obs_dir(self):
        # Get the direction to obstacle (in radian)
        dx = (self.agent.x + self.agent.W/2) - self.obs_x
        dy = (self.agent.y + self.agent.H/2) - self.obs_y
        if dy == 0.0:
            dy = 0.01 # Prevents Division by Zero Error
        theta = math.atan(dx/dy)

        # Adjust theta (range of atan(x) = (-pi/2, pi/2))
        if dy < 0:
            # if agent is located higher than obstacle
            if theta < 0:
                theta += math.pi
            else:
                theta -= math.pi

        return theta - self.agent.rotation

    def get_obs_dist(self):
        # Get the distance between agent and obstacle
        ax = self.agent.x + self.agent.W/2
        ay = self.agent.y + self.agent.H/2
        dsquare = (ax - self.obs_x)**2 + (ay - self.obs_y)**2
        return math.sqrt(dsquare) - 30

    def get_sim_state(self):
        sim_state =  np.array([
            (self.agent.y-self.PASS_H)/200.0,
            (self.agent.x-self.PATH_XPOS)/200.0,
            self.agent.rotation,
            self.get_obs_dist()/200.0,
            self.get_obs_dir()])#,
            #self.obs_r/200.0])

        return sim_state

    def step(self, action, expected_reward):
        pygame.event.pump()
        self.sim_ticks += 1

        # (1)
        # Update agent position based on 'action'(0:Turn Left, 1:Move Straight, 2:Move Right)
        if action == 0:
            self.agent.rotation += 0.08
        elif action == 2:
            self.agent.rotation -= 0.08
        else:
            pass


        # (2)
        # Update Obstacle position
        # self.obs_y += self.OBS_Vv

        
        # (3)
        # Update Screen
        self.screen.fill(self.COLOR_BLACK)
        # -> Finish line, center line
        pygame.draw.line(self.screen, self.COLOR_WHITE,
                [0, self.PASS_H], [self.SCREEN_W, self.PASS_H], 4)
        pygame.draw.line(self.screen, self.COLOR_PATH,
                [self.PATH_XPOS, 0], [self.PATH_XPOS, self.SCREEN_H], 4)
        # -> Agent
        self.agent.update()
        self.screen.blit(self.agent.image, (self.agent.x,self.agent.y))
        # -> Obstacle
        pygame.draw.circle(self.screen, self.COLOR_OBS, [self.obs_x, self.obs_y], self.obs_r)
        # -> Status window
        pygame.draw.rect(self.screen, self.COLOR_BLACK,
                pygame.Rect(0, self.SCREEN_H,
                            self.SCREEN_W, self.SCREEN_H + self.STATUS_H))
        pygame.draw.line(self.screen, self.COLOR_WHITE,
                [0, self.SCREEN_H], [self.SCREEN_W, self.SCREEN_H], 8)

        
        # (4)
        # Determine reward
        self.agent.step_reward = 0.0
        

        # if agent successfully avoided obstacle
        if self.agent.y < self.PASS_H:
            self.sim_over = True
            self.sim_over_why = '장애물 회피 성공'
            self.agent.win_count += 1
            self.agent.step_reward += 5.0
            
            # panelty proportional to distance to path
            self.agent.step_reward -= abs(self.agent.x - self.PATH_XPOS) / 200.0

            
        #print(self.agent.rect)
        # if agent collided with obstacle
        if self.get_obs_dist() < self.obs_r:
            self.sim_over = True
            self.sim_over_why = '장애물과 충돌'
            self.agent.step_reward = -5.0
        
        # if agent collided with walls
        if self.agent.x < 0 or self.agent.x + self.agent.W > self.SCREEN_W:
            self.sim_over = True
            self.sim_over_why = '경로 이탈'
            self.agent.step_reward = -5.0
            
        # if time exceeded 100 ticks
        if self.sim_ticks >= 100:
            self.sim_over = True
            self.sim_over_why = '시간 초과'
            self.agent.step_reward = -5.0

        self.agent.total_reward += self.agent.step_reward


        #Update sim_state
        self.sim_prev_state = self.sim_state
        self.sim_state = self.get_sim_state()

        # (5)
        # Update Status Monitor
        text = []
        text.append(self.font.render(
            "Agent Pos: ({0:.2f},{1:.2f}), Obstacle Pos: ({2:.2f},{3:.2f})".
            format(self.agent.x + self.agent.W/2, self.agent.y + self.agent.H/2, self.obs_x, self.obs_y),
                    True, self.COLOR_WHITE))
        text.append(self.font.render(
            "(State Vector) = [{0:.2f}, {1:.2f}, {2:.2f}˚, {3:.2f}, {4:.2f}˚]".format(
                self.sim_state[0], self.sim_state[1], self.sim_state[2]*180/math.pi,
                self.sim_state[3], self.sim_state[4]*180/math.pi),
                    True, self.COLOR_AGENT))
        text.append(self.font.render(
            "Step reward: {0:.3f}, Expected reward: {1:.3f}".
            format(self.agent.step_reward, expected_reward),
                    True, self.COLOR_WHITE))
        text.append(self.font.render(
            "Episode {0}, Tick {1}: {2}".
            format(self.episode_count, self.sim_ticks, self.sim_over_why),
                    True, self.COLOR_WHITE))
        for i in range(len(text)):
            self.screen.blit(text[i], (10, 10+self.SCREEN_H + self.FONT_SIZE*i))
        
        pygame.display.flip()
       

        self.clock.tick(300)

        if self.sim_over:
            time.sleep(0.1)
        

        return self.sim_state, self.agent.step_reward, self.sim_over
    
    def quit(self):
        pygame.quit()
