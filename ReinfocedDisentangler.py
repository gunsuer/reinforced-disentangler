import pennylane as qml
import pennylane.numpy as np
import gymnasium as gym

def Cliff2():
    """
    Random 2-qubit Clifford circuit.

    Arguments:
        -nodes (np.ndarray): 
    
    Returns:
        -null
    """
    
    weights = np.random.randint(2, size=(2, 10))
    
    return qml.matrix(qml.RandomLayers(weights=weights,wires=[0,1])).numpy()

def RandomLayers(N_QUBITS, DEPTH):
    """
    Generates brick wall pattern of random 2 qubit Clifford gates

    Arguments:
        -N_QUBITS (int): Number of qubits
        -DEPTH (int): Depth of the circuit

    Returns:
        -random_layers (np.ndarray): Array of 4x4 unitaries (N_QUBITS, DEPTH, 4, 4)
    
    """

    random_layers = []
    for t in range(DEPTH):
        layer = []
        for x in range(0,N_QUBITS,2):
                layer.append(Cliff2())
        random_layers.append(layer)

    return random_layers

def RandomFlip(theta,K):
    """
    Randomly flip K entries of a binary matrix theta.

    Arguments:
        -theta (np.ndarray):
        -K (int): 

    Returns:
        -flipped (np.ndarray):

    """
    
    x,y = np.shape(theta)
    N = int(x*y)
    arr = np.array([0] * (N-K) + [1] * K)
    np.random.shuffle(arr)
    arr = arr.reshape((x,y))

    flipped = (theta + arr) % 2
    
    return flipped

class RandomCircuit():
    """
    Random brick circuit model
    
    """

    def __init__(self, nqubits, depth,):
        self.nqubits = nqubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=nqubits)
                     

class Disentangler(gym.Env):
    """
    Reinforcement learning environment for the disentangler.

    """

    def __init__(self, random_layers):
        self.action_space = gym.spaces.Discrete(5)
        self.random_layers = random_layers
        self.nqubits = np.shape(random_layers)[0]
        self.depth = np.shape(random_layers)[1]
        self.state = np.zeros((self.nqubits,self.depth))
        self.moves = 20


    def step(self, action):
        self.state = RandomFlip(self.state, action)
        self.moves += -1

        entropies = RandomCircuit(self.state,self.random_layers)
        entropy = sum(entropies)/len(entropies)

        trivial1 = (sum(self.state[:,-1]) == len(self.state[:,-1]))
        trivial2 = (sum(self.state[:,-1]) == len(self.state[:,-1]) - 1)
        trivial = (trivial1 or trivial2)

        if self.moves <= 0:
            reward = 0
            done = True
        else:
            reward = 0
            done = False
        
        if entropy < 1e-20:
            reward = 100
            done = True

        if trivial:
            reward = -100
            done = True

        info = {}
        
        return self.state, reward, done, info
    
    def reset(self):
        self.state = np.zeros(np.shape(self.state))
        self.moves = 20

        return self.state