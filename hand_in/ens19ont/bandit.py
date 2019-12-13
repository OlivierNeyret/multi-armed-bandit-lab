# epsilon-greedy example implementation of a multi-armed bandit
import random
import statistics
import math

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import simulator
import reference_bandit

# generic epsilon-greedy bandit
class Bandit:
    def __init__(self, arms, epsilon=0.1):
        self.arms = arms.copy()
        self.epsilon = epsilon
        self.frequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.expected_values = [0] * len(arms)
        self.x = 0.98 # Decay value
        self.discarded = []
        self.threshold = 1

    def run(self):
        if min(self.frequencies) == 0:
            return self.arms[self.frequencies.index(min(self.frequencies))]
        if random.random() < self.epsilon:
            index = random.randint(0, len(self.arms) - 1)
            while(index in self.discarded):
                index = random.randint(0, len(self.arms) - 1)
            return self.arms[index]
        index = self.expected_values.index(max(self.expected_values))
        return self.arms[index]

    def give_feedback(self, arm, reward):
        arm_index = self.arms.index(arm)
        sum = self.sums[arm_index] + reward
        self.sums[arm_index] = sum
        frequency = self.frequencies[arm_index] + 1
        self.frequencies[arm_index] = frequency
        expected_value = sum / frequency
        self.expected_values[arm_index] = expected_value
        # Update epsilon
        self.epsilon *= self.x
        # Remove the worst arms (only if every arms have been tested a few times)
        # https://www.kaggle.com/nroman/detecting-outliers-with-chauvenet-s-criterion
        if min(self.frequencies) > 0:
            mean = statistics.mean(self.expected_values)
            stdev = statistics.stdev(self.expected_values)
            criterion = 1.0/(2*len(self.expected_values))
            distanceList = [] 
            for i in range(len(self.expected_values)):
                distanceList.append(math.erfc(abs(self.expected_values[i]-mean) / stdev))
            for i in range(len(distanceList)):
                if distanceList[i] * self.threshold < criterion and self.expected_values[i] < mean and len(self.expected_values) > 2 and not i in self.discarded: # Must be discard
                    print("ARM",i,"REMOVED")
                    self.discarded.append(i)
                    if len(self.discarded) == 1:
                        self.threshold = 0.22
                    if len(self.discarded) == 2:
                        self.threshold = 0.15
                    #del self.arms[i]
                    #del self.frequencies[i]
                    #del self.sums[i]
                    #del self.expected_values[i]
                    return # We stop otherwise it may try to discard another arm

# configuration
arms = [
    'Configuration a',
    'Configuration b',
    'Configuration c',
    'Configuration d',
    'Configuration e',
    'Configuration f'
]

# instantiate bandits
bandit = Bandit(arms)
ref_bandit = reference_bandit.ReferenceBandit(arms)
