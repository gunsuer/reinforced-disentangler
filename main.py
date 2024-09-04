# import setup.py
import itertools

sizes = list(itertools.product([2, 4], [2, 4]))

ts = 1e4
lr = 0.001
ec = 0.01

for size in sizes:
    n_qubits = size[0]
    depth = size[1]

    env = Disentangler(n_qubits=N_QUBITS, depth=DEPTH, random_layers=RandomLayers(N_QUBITS, DEPTH))
    env.reset()
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./tensorboard_files", learning_rate=lr, ent_coef=ec)

    model.learn(total_timesteps=ts, tb_log_name="{N_QUBITS}x{DEPTH}_tb_{ts}_{lr}_{ec}".format(N_QUBITS=N_QUBITS, DEPTH=DEPTH, ts=ts, lr=lr, ec=ec))
    model_path = "./models/{N_QUBITS}x{DEPTH}_PPO_{ts}_{lr}_{ec}".format(N_QUBITS=N_QUBITS, DEPTH=DEPTH, ts=ts, lr=lr, ec=ec)
    model.save(model_path)