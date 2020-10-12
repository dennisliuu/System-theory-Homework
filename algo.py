"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np

class EpislonGreedy(object):
    """
    Implementation of epislon-greedy algorithm.
    """

    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initialize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        assert (0. <= epislon <=
                1.0), "[ERROR] Epislon should be in range [0,1]"
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table. No need to return any result.
        """

        ################### Your code here #######################
        #raise NotImplementedError('[EpislonGreedy] update function NOT IMPLEMENTED')
        self._action_N[action] += 1
        self._Q[action] += (1/self._action_N[action]) * \
            (immi_reward - self._Q[action])
        ##########################################################

    def act(self, t):
        """
        Step 3: Choose the action via greedy or explore.
        Return: action selection
        """
        ################### Your code here #######################
        if np.random.random() > self._epislon:
            a = np.argmax(self._Q)
            return a
        else:
            a = np.random.randint(0, self._nb)
            return a
        #raise NotImplementedError('[EpislonGreedy] act function NOT IMPLEMENTED')
        ##########################################################


class UCB(object):
    """
    Implementation of upper confidence bound.
    """

    def __init__(self, NumofBandits=10, c=2):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._counts = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        #raise NotImplementedError('[UCB] update function NOT IMPLEMENTED')
        self._action_N[action] += 1
        self._Q[action] += (1/self._action_N[action]) * \
            (immi_reward - self._Q[action])
        ##########################################################

    def act(self, t):
        """
        Step 3: use UCB action selection. We'll pull all arms once first!
        HINT: Check out p.27, equation 2.8
        """
        ################### Your code here #######################
        #raise NotImplementedError('[UCB] act function NOT IMPLEMENTED')
        if t < self._nb:
            a = np.random.randint(self._nb)
            if self._action_N[a] == 0:
                return a
            else:
                return np.argmin(self._action_N)
        else:
            return np.argmax(self._Q + self._counts * np.sqrt(np.log(t) / self._action_N))
        ##########################################################


class Gradient(object):
    """
    Implementation of your gradient-based method
    """

    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._H = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        self._pi = np.zeros(self._nb, dtype=float)
        self._t = 0
        self._avg_reward = 0
        self._alpha = epislon
        for i in range(self._nb):
            self._pi[i] = 1/self._nb

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        self._avg_reward = self._avg_reward + \
            (immi_reward-self._avg_reward)/self._t
        for i in range(self._nb):
            if (i == action):
                self._H[i] = self._H[i]+self._alpha * \
                    (immi_reward-self._avg_reward)*(1-self._pi[i])
            else:
                self._H[i] = self._H[i]-self._alpha * \
                    (immi_reward-self._avg_reward)*self._pi[i]
        sum_h = 0
        for i in range(self._nb):
            sum_h += np.exp(self._H[i])
        for i in range(self._nb):
            self._pi[i] = np.exp(self._H[i])/sum_h
        ##########################################################

    def act(self, t):
        """
        Step 3: select action with gradient-based method
        HINT: Check out p.28, eq 2.9 in your textbook
        """
        ################### Your code here #######################
        # choose action
        self._t = t+1
        return np.random.choice(self._nb, p=self._pi)
        ##########################################################
