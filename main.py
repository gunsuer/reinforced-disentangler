import ReinfocedDisentangler
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

N_QUBITS = 2*3
DEPTH = 2

random_layers = ReinfocedDisentangler.RandomLayers(N_QUBITS,DEPTH)

env = ReinfocedDisentangler.Disentangler(random_layers=random_layers)

# test run

episodes = 2000
measurements = []
scores = []
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    
    measurements.append(n_state)
    scores.append(score)
    
    print('Episode:{} Score:{}'.format(episode, score))

# histogram for the efficient disentanglers
disentanglers = []
for i in range(len(scores)):
    if scores[i] == 100:
        disentanglers.append(measurements[i])

hashable_matrices = [tuple(map(tuple, matrix)) for matrix in disentanglers]

matrix_counts = Counter(hashable_matrices)

labels, counts = zip(*matrix_counts.items())

labels = [np.array(label) for label in labels]

plt.figure(figsize=(10,6))
plt.bar(range(len(counts)), counts, tick_label=["M_"+str(i+1) for i in range(len(labels))])
plt.xlabel('Matrices')
plt.ylabel('Count')
plt.title('Histogram of Matrix Occurrences')
plt.xticks(rotation=90)
plt.savefig('untitled.pdf')
