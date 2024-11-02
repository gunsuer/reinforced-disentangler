#!/bin/bash
#SBATCH --job-name=qip				                      # Job name
#SBATCH --output=output_%j.log		                # Output log file (%j expands to jobID)
#SBATCH --error=error_%j.log		                  # Error log file (%j expands to jobID)
#SBATCH --time=12:00:00 		                      # Time limit (hh:mm:ss)
#SBATCH --nodes=1		                             # Number of nodes
#SBATCH --ntasks=1		                            # Number of tasks (generally set to 1 for a single script)
#SBATCH --cpus-per-task=50			                   # Number of CPU cores per task
#SBATCH --mem=256GB                             # Memory per node
#SBATCH --partition=short                       # Partition/queue to submit to
#SBATCH --mail-user=suer.g@northeastern.edu     # Email notifications
#SBATCH --mail-type=ALL

# Load any required modules
conda init bash
conda activate miniconda3/envs/qip              #Load the Python module (modify version as needed)

# Run the Python script

srun python main.py -n 10 -d 2 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 3 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 5 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 8 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 10 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 12 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 15 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 20 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 25 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50
srun python main.py -n 10 -d 30 -ts 500000 -lr 1e-3 -ec 1e-2 -aa 1 -pr 50

# Deactivate conda
conda deactivate
