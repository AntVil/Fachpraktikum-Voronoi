import os

# MARK: File System
DATA_FOLDER: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# Number of repetitions for the benchmark analysis
RUNS: int = 20


# info = np.iinfo(np.int32)
# print("Minimum:", info.min)  # -2147483647
# print("Maximum:", info.max)  #  2147483647
INT32_MAX: int = 2147483647
