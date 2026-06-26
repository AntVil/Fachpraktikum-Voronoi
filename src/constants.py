import os

# MARK: File System
DATA_FOLDER: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# Number of repetitions for the benchmark analysis
RUNS: int = 20


# info = np.iinfo(np.int64)
# print("Minimum:", info.min)  # -9223372036854775808
# print("Maximum:", info.max)  #  9223372036854775807
INT64_MAX: int = 9223372036854775807
