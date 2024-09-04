import sys
sys.path.insert(0, '..')
import numpy as np
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import pyclifford as pc
import torch
import statistics
from numba import njit
import seaborn as sns
sns.set_theme()


def one_layer_brick_wall(circ,even=True):
    """
    One layer brick wall of random Clifford circuits
    """
    for ii in range(int(circ.N//2)):
        if even:
            circ.gate(2*ii,2*ii+1)
        else:
            circ.gate((2*ii+1),(2*ii+2)%circ.N)


    return circ

def theta(N_QUBITS,DEPTH):
    """

    Arguments:
        number of qubits, depth
    
    Returns:
        Random 2-qubit Clifford gates for a single layer.
        
    """
    
    theta= np.random.randint(2, size=(DEPTH,N_QUBITS))


    return theta

def one_layer_measurement(circ,the,layer):
    """
    Input:
        
    """
     
    positions=[]
    for i in range(0,circ.N):
        if the[layer,i]==int(1):
            positions.append(int(i))
        else:
            continue
        
    if positions!=[]:
        circ.measure(*positions)


    return circ

def create_circuit(N_QUBITS,DEPTH,the):
    circ = pc.circuit.Circuit(N_QUBITS)
    for i in range(DEPTH):
        circ = one_layer_brick_wall(circ,even=True)
        #circ = one_layer_measurement(circ,the,int(2*DEPTH-1-2*i))
        circ = one_layer_brick_wall(circ,even=False)
        #circ = one_layer_measurement(circ,the,int(2*DEPTH-1-2*i-1))


    return circ

def averaged_EE(state_final):
    EE_positions=[]
    for i in range(0,N_QUBITS):
        EE_positions.append([int(i),int(i+1)%N_QUBITS])
        
    EE_list=[]
    for i in range(0,len(EE_positions)):
        EE_list.append(state_final.entropy(EE_positions[i]))
    
    averaged_EE=statistics.mean(EE_list)

        
    return averaged_EE, EE_positions, EE_list

class Disentangler(gym.Env):
    """
    Reinforcement learning environment for the disentangler.
    """
    
    def __init__(self, n_qubits, depth, random_layers):
        super(Disentangler, self).__init__()
        
        self.N_QUBITS = n_qubits
        self.DEPTH = depth
        self.random_layers = random_layers

        self.action_space = gym.spaces.Discrete(self.N_QUBITS * self.DEPTH)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.N_QUBITS, self.DEPTH), dtype=np.int8)
        self.state = np.zeros((self.N_QUBITS, self.DEPTH), dtype=np.int8)

    def step(self, action):
        # Initialize reward and done
        reward = 0
        done = False
        truncate = False

        # Apply the action
        h = np.zeros(self.N_QUBITS * self.DEPTH, dtype=np.int8)
        h[action] = 1
        h = h.reshape((self.N_QUBITS, self.DEPTH))
        self.state = (self.state + h) % 2

        # Calculate entropy (assumes circuit is a predefined function)
        entropies = circuit(self.state, self.random_layers)
        entropy = np.mean(entropies)

        # Check if the state is trivial
        non_trivial = any(self.state[:,-1][i] == 0 and self.state[:,-1][(i + 1) % self.N_QUBITS] == 0 for i in range(self.N_QUBITS))

        # Determine reward and done conditions
        if entropy < 1e-15 and non_trivial:
            reward = 100
            done = True
        
        # Return the state, reward, done flag, truncate flag, and info
        info = {}
        return self.state, reward, done, truncate, info
    
    def reset(self, seed=None):
        # Seed the random number generator if a seed is provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the state to an all-zero matrix
        self.state = np.zeros((self.N_QUBITS, self.DEPTH), dtype=np.int8)

        info = {}
        return self.state, info
    
    def render(self):
        print()

    def close(self):
        # Optional: Implement any cleanup
        pass