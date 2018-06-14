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
plt.style.use('ggplot')

class SarsaAgent():
    """A sarsa agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, temperature=0.1, trace_decay_rate = 0.95, grid_size=20, num_actions = 3, learning_rate = 1e-2, weight_init='constant', weight=0.5):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car
        
        if weight_init == 'uniform':
          self.network_weights = np.random.rand(num_actions,grid_size*grid_size)
        elif weight_init == 'constant':
          self.network_weights = weight*np.ones((num_actions,grid_size*grid_size))
        self.activations = np.zeros((grid_size,grid_size))
        self.activations1 = np.zeros((grid_size,grid_size))
        
        xcenters = np.linspace(-150,30,grid_size+1,False)[1:]
        xdcenters = np.linspace(-15,15,grid_size+1,False)[1:]
        self.centers = np.array(np.meshgrid(xcenters,np.flip(xdcenters,axis=0)))
#         self.centers = np.array(np.meshgrid(xcenters,xdcenters))
        
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
        

    def learn_task(self, n_episodes = 200, n_steps = 2000):
        """Run trials and learn, without display.

        Parameters
        ----------
        n_episodes -- number of episodes
        n_steps -- number of steps to simulate for each episode
        """
        
        steps_till_success = np.zeros(n_episodes)
        for i in range(n_episodes):
            
            # make sure the mountain-car is reset
            self.mountain_car.reset()
            self.eligibility_traces.fill(0.0)
            
            # simulate the first timestep
            q, self.activations = self.action_values()
            p = np.exp(q/self.temperature)/sum(np.exp(q/self.temperature))
            # choose the action with a softmax
            self.a = np.random.choice(3,p=p)
            self.q = q[self.a]
    
            for n in range(n_steps):
                
                # simulate the timestep
                self.mountain_car.apply_force(self.a-1)
                self.mountain_car.simulate_timesteps(100, 0.01)
                self.r = self.mountain_car.R
                q1, self.activations1 = self.action_values()
                p = np.exp(q1/self.temperature)/sum(np.exp(q1/self.temperature))
                
                # choose the action with a softmax
                self.a1 = np.random.choice(3,p=p)
                self.q1 = q1[self.a1]
                
                #learning step
                self.learn()
                self.a = self.a1
                self.activations = self.activations1
                self.q = self.q1
                
                # check for rewards
                if self.mountain_car.R > 0.0:
                    break
            
#             print('\rEpisode ' + str(i+1) + " reward obtained at t = " + str(self.mountain_car.t))    
            steps_till_success[i] = self.mountain_car.t
        
        return steps_till_success

    def action_values(self, x = None, x_d = None):
        
        if x is None:
            x = self.mountain_car.x
        if x_d is None:
            x_d = self.mountain_car.x_d
        
        activations = ((self.centers[0] - x)/self.sigma[0])**2
        activations += ((self.centers[1] - x_d)/self.sigma[1])**2
        activations = np.exp(-activations)
        
        output = np.matmul(self.network_weights,activations.flatten())
        
        return output, activations

    def learn(self):
        delta = self.r + self.discount_factor*self.q1 - self.q
        
        self.eligibility_traces *= self.discount_factor*self.trace_decay_rate
        self.eligibility_traces[self.a,:] += self.activations.flatten()
        
        self.network_weights += self.learning_rate*delta*self.eligibility_traces

    def visualize_policy(self):
        labels = np.zeros_like(self.activations)
        
        for ix,iy in np.ndindex(self.activations.shape):
            q, _ = self.action_values(x=self.centers[0,ix,iy],x_d=self.centers[1,ix,iy])
            labels[ix,iy] = np.argmax(q)-1
        
        xcenters = self.centers[0].flatten()
        xdcenters = self.centers[1].flatten()
        labels = labels.flatten()
        
        plt.figure()
        plt.quiver(xcenters,xdcenters,labels,np.zeros_like(labels),np.arctan2(np.zeros_like(labels), labels),pivot='middle', cmap = 'Dark2')
        plt.xlabel('Position',fontsize=15)
        plt.ylabel('Velocity',fontsize=15)
        
    
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
            sys.stdout.flush()
            
            # choose a random action
            q, self.activations = self.action_values()
            p = np.exp(q/self.temperature)/sum(np.exp(q/self.temperature))
            # choose the action with a softmax
            self.a = np.random.choice(3,p=p)
            self.mountain_car.apply_force(self.a-1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()            
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break


n_episodes = 100
latency = np.zeros((10,n_episodes))
for i in range(10):
    print('Trial ', i)
    d = SarsaAgent(weight_init = 'constant', learning_rate=0.01, temperature=0.01, trace_decay_rate = 0.99, grid_size=20)
    latency[i,:] = d.learn_task(n_episodes = n_episodes, n_steps = 3000)
  
plt.plot(np.mean(latency,axis=0),linewidth=2.0)
plt.grid()
plt.xlabel('Episodes',fontsize=15)
plt.ylabel('Average Escape Latency',fontsize=15)
d.visualize_trial()
plb.show()


#Question 2
d = SarsaAgent(weight_init = 'constant', learning_rate=0.01, temperature=0.01, trace_decay_rate = 0.99, grid_size=20)
d.visualize_policy()
for i in range(int(n_episodes/25)):
  d.learn_task(n_episodes = 25, n_steps = 3000)
  d.visualize_policy()
  
#Question 3
temperature_range = [0.01,0.1,1.0]
n_episodes = 100
latency = np.zeros((4,n_episodes))

for j,t in enumerate(temperature_range):
  local_variable = np.zeros((5,n_episodes))
  for i in range(5):
    print('Temperature ', t, ' Trial ',i)
    d = SarsaAgent(weight_init = 'constant', learning_rate=0.01, temperature=t, trace_decay_rate = 0.99, grid_size=20)
    local_variable[j,:] = d.learn_task(n_episodes = n_episodes, n_steps = 3000)
  latency[j,:] = np.mean(local_variable,axis=0)
  
plt.plot(latency[:-2].T)
plt.legend(['Temperature = 0.01','Temperature = 0.1'])
plt.xlabel('Episodes')
plt.ylabel('Escape latency')