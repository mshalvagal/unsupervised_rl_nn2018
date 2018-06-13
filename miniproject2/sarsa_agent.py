#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:57:35 2018

@author: manu
"""

import sys

import pylab as plb
import numpy as np
import mountaincar
import matplotlib.pyplot as plt

class SarsaAgent():
    """A sarsa agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, temperature=1.0, trace_decay_rate = 0.95, grid_size=20, num_actions = 3, learning_rate = 1e-4):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.network_weights = np.random.rand(num_actions,grid_size*grid_size)
        self.activations = np.zeros((grid_size,grid_size))
        self.activations1 = np.zeros((grid_size,grid_size))
        
        xcenters = np.linspace(-150,30,grid_size)
        xdcenters = np.linspace(-15,15,grid_size)
        self.centers = np.array(np.meshgrid(xcenters,np.flip(xdcenters,axis=0)))
        
        self.sigma = np.zeros(2)
        self.sigma[0] = xcenters[1]-xcenters[0]
        self.sigma[1] = xdcenters[1]-xdcenters[0]
        
        self.temperature = temperature
        self.discount_factor = 0.95
        self.learning_rate = learning_rate
        self.trace_decay_rate = trace_decay_rate
        self.eligibility_traces = np.zeros_like(self.network_weights)
        
        self.a = 0.0
        self.a1 = 0.0
        self.r = 0.0
        self.q = 0.0
        self.q1 = 0.0
        

    def learn_task(self, n_episodes = 2000, n_steps = 200):
        """Run trials and learn, without display.

        Parameters
        ----------
        n_episodes -- number of episodes
        n_steps -- number of steps to simulate for each episode
        """
        
        steps_till_success = np.zeros(n_episodes)
        for i in range(n_episodes):
            # make sure the mountain-car is reset
            print('\rEpisode ', i+1)
            
            self.mountain_car.reset()
            self.eligibility_traces.fill(0.0)
            
            # simulate the first timestep
            q, self.activations = self.action_values()
            # choose the action with a softmax
            self.a = np.random.choice(3,p=q)-1
            self.q = q[self.a+1]
    
            for n in range(n_steps):
                
                # simulate the timestep
                self.mountain_car.apply_force(self.a)
                self.mountain_car.simulate_timesteps(100, 0.01)
                self.r = self.mountain_car.R
                q1, self.activations1 = self.action_values()
                
                # choose the action with a softmax
                self.a1 = np.random.choice(3,p=q1)-1
                self.q1 = q1[self.a1+1]
                
                #learning step
                self.learn()
                self.a = self.a1
                self.activations = self.activations1
                self.q = self.q1
                
                # check for rewards
                if self.mountain_car.R > 0.0:
                    print("\rreward obtained at t = ", self.mountain_car.t)
                    break
                
            steps_till_success[i] = self.mountain_car.t
        
        return steps_till_success

    def action_values(self):
        activations = (self.centers[0] - self.mountain_car.x)**2/self.sigma[0]
        activations += (self.centers[1] - self.mountain_car.x_d)**2/self.sigma[1]
        activations = np.exp(-activations)
        
        output = np.dot(self.network_weights,activations.flatten())/self.temperature
        output = np.exp(output)/sum(np.exp(output))
        
        return output, activations

    def learn(self):
        delta = self.r + self.discount_factor*self.q1 - self.q
        self.network_weights += self.learning_rate*delta*self.eligibility_traces
        
        self.eligibility_traces *= self.discount_factor*self.trace_decay_rate
        self.eligibility_traces[self.a+1,:] += self.activations.flatten()

    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            sys.stdout.flush()
            
            # choose a random action
            q, self.activations = self.action_values()
            # choose the action with a softmax
            self.a = np.random.choice(3,p=q)-1
            self.mountain_car.apply_force(self.a)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()            
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

if __name__ == "__main__":
    d = SarsaAgent()
    steps = d.learn_task()
    plt.plot(steps)
    d.visualize_trial()
    plb.show()
