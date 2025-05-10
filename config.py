# Configuration file for the project
PATH_MODELS = "models"
DEVICE = "cuda"


MODEL = "llava" # Possible values: qwen, llava
STEP_DISTANCE = 1 # Distance between steps to create pairs

PATH_DATASET = "dataset/english_N1000_2025-02-11.json"
#PATH_DATASET = "dataset/italian_N50_2024-11-29.json"
# PATH_DATASET = "dataset/italian_N1000_2025-02-11.json"
#PATH_DATASET = "dataset/italian_from_eng_N1000_2025-02-11.json"

PROMPT = "Does the step shown in the first image happen after the step in the second? Answer with 'yes' or 'no'."
TYPE_PROMPT = "after" # Possible values: before, after
RANDOM_ORDER = True # If True, the order of the steps will be random



