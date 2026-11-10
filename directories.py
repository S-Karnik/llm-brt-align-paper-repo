import os

PROJECT_ROOT_DIR = os.environ.get("LLM_REACHABILITY_PROJECT_ROOT", os.getcwd())
OUTPUT_ROOT_DIR = os.environ.get("LLM_REACHABILITY_OUTPUT_ROOT", PROJECT_ROOT_DIR)
DATA_CACHE_DIR = os.environ.get(
    "LLM_REACHABILITY_DATA_ROOT",
    os.path.join(PROJECT_ROOT_DIR, "data_cache"),
)
TRAINED_VALUE_FUNCTION_MODELS_DIR = os.path.join(OUTPUT_ROOT_DIR, 'trained_value_function_models')

# Where to save the ground truth labels for the LLM generated datasets
GROUND_TRUTH_PER_TOKEN_LABELS_DIR = os.path.join(OUTPUT_ROOT_DIR, "ground_truth_per_token_labels")
