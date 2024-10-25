#!/bin/bash
#SBATCH --job-name=qip                   # Job name
#SBATCH --output=output_%j.log           # Output log file (%j expands to jobID)
#SBATCH --error=error_%j.log             # Error log file (%j expands to jobID)
#SBATCH --time=01:00:00                  # Time limit (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (generally set to 1 for a single script)
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=4GB                        # Memory per node
#SBATCH --partition=standard             # Partition/queue to submit to

# Load any required modules
module load python/3.8                   # Load the Python module (modify version as needed)

# Activate a virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the Python script
srun python your_script.py               # Replace 'your_script.py' with your Python script
