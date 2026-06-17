import os

# MARK: File System
DATA_FOLDER: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# Number of repetitions for the benchmark analysis
RUNS: int = 20
