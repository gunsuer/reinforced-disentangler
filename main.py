# ------------------------------------------------------------------------------
# Efficient Disentanglers with Proximal Policy Optimization
# 
# 
# 10/25/2024
# ------------------------------------------------------------------------------

# Import packages
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyclifford as pc
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import pyclifford as pc
from pyclifford.utils import mask, condense, pauli_diagonalize1,stabilizer_measure
from pyclifford.paulialg import Pauli, pauli, PauliMonomial, pauli_zero
from pyclifford.stabilizer import (StabilizerState, CliffordMap, zero_state, identity_map, clifford_rotation_map, random_clifford_map)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='This script does something with arguments.')

# Add arguments
parser.add_argument('-n', '--qubits', type=int, help='Number of qubits')
parser.add_argument('-d', '--depth', type=int, help='Circuit depth')
parser.add_argument('-ts', '--ts', type=int, help='Total time steps')
parser.add_argument('-lr', '--lr', type=float, help='Learning rate')
parser.add_argument('-ec', '--ec', type=float, help='Entropy coefficient')
parser.add_argument('-aa', '--alpha', type=float, help='Penalty slope')
parser.add_argument('-pr', '--positivereward', type=float, help='Positive reward')

# Parse the arguments
args = parser.parse_args()

# Function definitions
def random_layers(N_QUBITS,HALF_DEPTH):
    """
    Random layers
    """

    random_layers=[]
    for i in range (int(HALF_DEPTH)):
        random_layer=[]
        if i%2==0:
            for j in range (int(np.floor(N_QUBITS/2))):
                gate=pc.CliffordGate(j*2,j*2+1)
                gate.set_forward_map(pc.random_clifford_map(2))
                random_layer.append(gate)
            random_layers.append(random_layer)
        elif i%2==1:
            for j in range (int(np.floor(N_QUBITS/2))):
                gate=pc.CliffordGate(j*2+1,(j*2+2)%N_QUBITS)
                gate.set_forward_map(pc.random_clifford_map(2))
                random_layer.append(gate)
            random_layers.append(random_layer)
    return random_layers

def measure_layers(N_QUBITS,HALF_DEPTH,theta):
    """
    Measurement layers
    """ 

    measure_layers=[]
    for i in range (int(HALF_DEPTH)):
        measure_layer=[]
        for j in range (int(N_QUBITS)):
            if theta[-i + HALF_DEPTH - 1][j]==1:
                measure_layer.append(j)
        measure_layers.append(measure_layer)
    return measure_layers

def create_circuit(N_QUBITS,HALF_DEPTH,random_layers,measure_layers):
    """
    Create brickwall circuit
    """

    circ = pc.circuit.Circuit(N_QUBITS)
    for i in range(int(HALF_DEPTH)):
        for j in range(int(np.floor(N_QUBITS/2))):
            circ.take(random_layers[i][j])
        if measure_layers[i]!=[]:
            qubits=tuple(measure_layers[i])
            circ.measure(*qubits)
            
    return circ

def averaged_EE(state_final):
    """
    Averaged von Neumann entropy
    """

    EE_positions=[]
    for i in range(0,state_final.N):
        EE_positions.append([int(i),int(i+1)%state_final.N])
        
    EE_list=[]
    for i in range(0,len(EE_positions)):
        EE_list.append(state_final.entropy(EE_positions[i]))
    
    averaged_EE=np.mean(EE_list)
        
    return averaged_EE

def penalty(x,penalty_slope):
    """
    Measurement cost function
    """

    return 2*(0.5-(1/(1+np.exp(-penalty_slope*x))-0.5))

class Disentangler(gym.Env):
    """
    Reinforcement learning environment for the disentangler.
    """
    
    def __init__(self, n_qubits, half_depth, positive_reward, penalty_slope):
        super(Disentangler, self).__init__()
        
        self.N_QUBITS = n_qubits
        self.HALF_DEPTH = half_depth
        self.DEPTH = 2*half_depth
        self.positive_reward = positive_reward
        self.penalty_slope = penalty_slope
        
        self.action_space = gym.spaces.Discrete(self.N_QUBITS * self.HALF_DEPTH)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.HALF_DEPTH, self.N_QUBITS), dtype=np.int8)

        self.random_layers=random_layers(self.N_QUBITS,self.HALF_DEPTH)
        self.theta = np.zeros((self.HALF_DEPTH, self.N_QUBITS), dtype=np.int8)
        self.measure_layers=measure_layers(self.N_QUBITS,self.HALF_DEPTH,self.theta)
        
    def step(self, action):
        # Initialize reward and done
        reward = 0
        done = False
        truncate = False

        # Apply the action
        h = np.zeros(self.N_QUBITS * self.HALF_DEPTH, dtype=np.int8)
        h[action] = 1
        h = h.reshape((self.HALF_DEPTH, self.N_QUBITS))
        self.theta = (self.theta + h) % 2

        # Calculate entropy (assumes circuit is a predefined function)
        self.measure_layers=measure_layers(self.N_QUBITS, self.HALF_DEPTH, self.theta)
        circ = create_circuit(self.N_QUBITS, self.HALF_DEPTH, self.random_layers, self.measure_layers)

        state_initial = pc.stabilizer.zero_state(self.N_QUBITS)
        state_final = circ.forward(state_initial)

        entropy = averaged_EE(state_final)

        
        if entropy == 0:
            m_per_layer = [np.sum(layer) for layer in self.theta]
            cost = [m_per_layer[i]*penalty(i,self.penalty_slope) for i in range(self.HALF_DEPTH)]
            penalty_list=[penalty(i,self.penalty_slope) for i in range(self.HALF_DEPTH)]
            reward = 1 - sum(cost)/(self.N_QUBITS*sum(penalty_list))
            #reward = self.positive_reward - np.sum(cost)/(self.N_QUBITS*self.DEPTH)
            done = True
        
        # Return the state, reward, done flag, truncate flag, and info
        info = {}
        return self.theta, reward, done, truncate, info
    
    def return_entropy(self):
        # Calculate entropy (assumes circuit is a predefined function)
        self.measure_layers=measure_layers(self.N_QUBITS, self.HALF_DEPTH, self.theta)
        circ = create_circuit(self.N_QUBITS, self.HALF_DEPTH, self.random_layers, self.measure_layers)

        state_initial = pc.stabilizer.zero_state(self.N_QUBITS)
        state_final = circ.forward(state_initial)

        entropy = averaged_EE(state_final)
        
        return entropy
        
    
    def reset(self, seed=None):
        # Seed the random number generator if a seed is provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the state to an all-zero matrix
        self.theta = np.zeros((self.HALF_DEPTH, self.N_QUBITS), dtype=np.int8)

        # Reset the random layers to another random layers
        self.random_layers=random_layers(self.N_QUBITS,self.HALF_DEPTH)
        
        info = {}
        return self.theta, info
    
    def render(self):
        print()

    def close(self):
        # Optional: Implement any cleanup
        pass

# ------------------------------------------------------------------------------

N_QUBITS = args.qubits
HALF_DEPTH = args.depth
DEPTH = 2*HALF_DEPTH
ts = args.ts
lr = args.lr
ec = args.ec
positive_reward = args.positivereward
penalty_slope = args.alpha

env = Disentangler(N_QUBITS,HALF_DEPTH,positive_reward,penalty_slope)

env.reset()
model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./tensorboard_files", learning_rate=lr, ent_coef=ec)

model.learn(total_timesteps=ts, tb_log_name="{N_QUBITS}x{HALF_DEPTH}_tb_{ts}_pr_{positive_reward}_ps_{penalty_slope}".format(N_QUBITS=N_QUBITS, HALF_DEPTH=HALF_DEPTH, ts=ts, positive_reward=positive_reward, penalty_slope=penalty_slope))

model_path = "./models/{N_QUBITS}x{HALF_DEPTH}_tb_{ts}_pr_{positive_reward}_ps_{penalty_slope}".format(N_QUBITS=N_QUBITS, HALF_DEPTH=HALF_DEPTH, ts=ts, positive_reward=positive_reward, penalty_slope=penalty_slope)

model.save(model_path)
