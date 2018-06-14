import pylab as plb
import matplotlib.pyplot as plt
import numpy as np
import math
import mountaincar


class S():
    # class for state (or center)

    def __init__(self, x, psi):
        self.x = x
        self.psi = psi

    def t(self):
        return (self.x, self.psi)


class SARSAAgent():

    def __init__(self, mountain_car = None, eta=0.01, tau=lambda x: 0.01, lambd=0.98, weight=0.5, seed=11):
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        ### SARSA parameters
        self.tau = tau # temperature -> exploration vs exploitation parameter.
        self.eta = eta # learning rate for weight update << 1
        self.lambd = lambd # eligibility decay rate 0 < lambda < 1
        self.gamma = 0.95 # reward factor
        self.weights_hist = [[],[]]

        ### setting up the neural network
        # computing interval parameters
        self.nNeuronsX = 20 # minimum 2
        self.nNeuronsPsi = 20
        self.inputDim = self.nNeuronsX * self.nNeuronsPsi
        self.outputDim = 3
        self.xCenters = np.linspace(-150, 30, self.nNeuronsX+1, False)[1:] # split the position interval excluding extreme values
        self.psiCenters = np.linspace(-15, 15, self.nNeuronsPsi+1, False)[1:] # split the speed interval excluding extreme values
        self.sigmaX = self.xCenters[1] - self.xCenters[0]
        self.sigmaPsi = self.psiCenters[1] - self.psiCenters[0]

        # generating the input neurons
        iN = []
        for psic, psi in enumerate(self.psiCenters):
            iN.append([])
            for xc, x in enumerate(self.xCenters):
                iN[psic].append(S(x, psi))
        self.iN = np.transpose(np.array(iN))

        # listing the actions
        self.ActDict = [-1,0,1]

        np.random.seed(seed)

        self.weights = np.ones((3, self.iN.shape[0], self.iN.shape[1])) * weight

        self.eligibilities = np.zeros((3, self.iN.shape[0], self.iN.shape[1]))        

    def one_run(self, n_steps = 2000, visual_no_learn = False):

        # if visualisation on, init figure
        if visual_no_learn:
            plb.ion()
            mv = mountaincar.MountainCarViewer(self.mountain_car)
            mv.create_figure(n_steps, n_steps)
            plb.draw()
            plb.pause(0.001)

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        # 1)Being in state s choose action a according to policy
        # replace state by r values (the response of the neurons)
        self.state = S(self.mountain_car.x, self.mountain_car.x_d)
        self.state_response = self.compute_response(self.state)
        self.action = self.choose_action(self.state_response, self.tau(0))

        self.eligibilities = np.zeros((3, self.iN.shape[0], self.iN.shape[1]))   

        for n in range(n_steps):

            # 2) Observe reward r and next state s'
            self.mountain_car.apply_force(self.ActDict[self.action])
            self.mountain_car.simulate_timesteps(100, 0.01)

            self.new_state = S(self.mountain_car.x, self.mountain_car.x_d)
            self.new_state_response = self.compute_response(self.new_state)

            # 3) Choose action a' in state s' according to policy
            self.new_action = self.choose_action(self.new_state_response, self.tau(n + 1))

            if visual_no_learn:
                # update the visualization
                mv.update_figure()
                plb.draw()
                plb.pause(0.001)
            else:
                # 4) Update weights
                self.weights_hist[0].append(self.weights.mean())
                self.weights_hist[1].append(self.weights.std())  # save the evolution of the weight

                self.eligibilities[self.action] += self.state_response
                delta = self.mountain_car.R + self.gamma * self.Q(self.new_state_response, self.new_action) - self.Q(self.state_response, self.action)
                self.weights = self.weights + self.eta * delta * self.eligibilities
                self.eligibilities *= self.gamma * self.lambd
                
            # 5) s' -> s; a' -> a
            self.state = self.new_state
            self.state_response = self.new_state_response

            self.action = self.new_action

            # check for rewards
            if self.mountain_car.R > 0.0:
                return n

        return n_steps


    def nRuns(self, runs = 100, max_steps = 2000, view_last = False):
        i = 0
        self.steps = []

        # signal to stop to avoid non convergence
        stop = False
        past = 0
            
        while i < runs :

            # if not convering to good result, stop to save time
            if not stop and i > 20:
                past = 0
                for j in range(20):
                    past += self.steps[-j]
                past /= 20

                if past > 3000:
                    stop = True

            if not stop:
                # run once if converging
                self.steps.append(self.one_run(max_steps))
            else:
                self.steps.append(past)

            # increment counter and display steps
            i += 1
            print("Run : {:d}, steps = {:d}".format(i, self.steps[-1]))

        # display a supplementary run graphically
        if view_last:
            self.one_run(80, True)

    # choose the action according to the probability
    def choose_action(self, state_response, tau):
        Ps = self.softmax(state_response, tau)
        # choose action
        rnd = np.random.uniform(low=0.0, high=1.0)
        action = len(Ps)-1
        for a in range(len(Ps)-1):
            if rnd < Ps[a]:
                action = a
                break
            else:
                rnd -= Ps[a]
        return action

    # compute the probability using softmax
    def softmax(self, state_response, tau=0.01):
        Ps = [0 for i in range(3)]  # will hold probabilities to take action
        Z = 0  # P normalization accumulator
        for a in range(3):
            Ps[a] = np.exp(self.Q(state_response, a) / tau)  # numerators of final Ps
            Z += Ps[a]
        return [P / Z for P in Ps]  # normalizing the probabilities

    # get the state-action value
    def Q(self, state_response, action):
        return np.multiply(self.weights[action], state_response).sum()

    def compute_response(self, state):
        response = np.zeros(self.iN.shape)
        for x in range(response.shape[0]):
            for psi in range(response.shape[1]):
                response[x][psi] = np.exp(-((self.iN[x][psi].x-state.x)/self.sigmaX)**2 -((self.iN[x][psi].psi-state.psi)/ self.sigmaPsi)**2)
        return response


if __name__ == "__main__":

    # code for single run

    d = SARSAAgent(eta=0.01, tau=lambda x:0.01, lambd=0.98, weight=0.5, seed=10)
    d.nRuns(200, 5000, True)

    # code for generating figures
    
    # params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    # plt.rcParams.update(params)
    
    # plt.figure(figsize=(8,5))
    # plt.title('Learning Curve')

    # runs = 200
    # agents = 1

    # # for lambd in [1, 0.98, 0.9, 0.5]:
    # # tau_name = [r'Decreasing $\tau$', r'$\tau$ = 0.1']
    # # for idx, tau in enumerate([lambda x: max(0.1*math.exp(-x/50.0), 0.01),lambda x: 0.1]):
    # for w in [0, 0.5, 1, 2]:
    #     summary = []
    #     for i in range(agents):
    #         d = SARSAAgent(seed=i, weight=w)
    #         d.nRuns(runs, 5000, False)
    #         summary.append(d.steps)

    #     y = []
    #     for i in range(runs):
    #         s = 0.0
    #         for j in range(agents):
    #             s += summary[j][i]
    #         s /= agents
    #         y.append(s)

    #     # print y
    #     # print d.weights_hist[0]
    #     # print d.weights_hist[1]

    #     # plt.plot([i for i in range(runs)], y, label=r'$\lambda = {}$'.format(lambd))
    #     # plt.plot([i for i in range(runs)], y, label=tau_name[idx])
    #     # plt.plot([i for i in range(runs)], y, label=r'$Default$')
    #     # plt.plot([i for i in range(runs)], y, label=r'$W = {}$'.format(w))
    #     plt.plot([i for i in range(len(d.weights_hist[0]))], d.weights_hist[0], label=r'$W = {}$'.format(w))


    # # plt.ylabel('Average escaping latency')
    # plt.ylabel('Mean value of weight matrix')
    # plt.xlabel('Number of episodes')

    # plt.legend(loc=1)

    # plt.savefig('./weight_mean.png', dpi=90, bbox_inches='tight')
    

    # code for analysis
    '''
    xSamples = np.linspace(-150, 30, 10+1, False)[1:]
    psiSamples = np.linspace(-15, 15, 10+1, False)[1:]

    quiverX = []
    quiverY = []
    quiverY0 = []
    quiverX1 = []
    quiverY2 = []
    for j, psiSample in enumerate(psiSamples):
        for i, xSample in enumerate(xSamples):
            state = d.compute_response(S(xSample, psiSample))
            qs = d.softmax(state, 0.01)
            quiverX.append(i)
            quiverY.append(j)
            quiverY0.append(qs[0])
            quiverX1.append(qs[1])
            quiverY2.append(qs[2])

    zeros = np.zeros(len(quiverX))
    quiverY0 = -1 * np.array(quiverY0)

    #labels
    plb.figure(7)
    plb.clf()
    ax = plb.axes()

    ax.set_xticklabels(['{:.1f}'.format(a) for a in xSamples])
    ax.set_yticklabels(['{:.1f}'.format(a) for a in psiSamples])
    plb.xticks(np.arange(len(xSamples))+0.5)
    plb.yticks(np.arange(len(psiSamples))+0.5)

    # different visualization
    # plb.quiver(quiverX, quiverY, quiverY0, zeros, scale=2, units='y')
    # plb.quiver(quiverX, quiverY, zeros, quiverX1, scale=2, units='y')
    plb.quiver(quiverX, quiverY, quiverY2, zeros, scale=2, units='y')

    plb.show()
    '''
