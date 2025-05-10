import os
from config import *
from function import evaluate_predictions, evaluate_consistency_predictions

if __name__ == "__main__":

    name_dataset = "italian_N50_2024-11-29"
    n_steps = 5
    prompt_before = "002"
    prompt_after = "002"
    name_model = "llava"

    path_before = os.path.join(
        f"results_{name_model}",
        name_dataset,
        "before",
        f"{n_steps}_steps",
        prompt_before,
        "results.csv",
    )

    path_after = os.path.join(
        f"results_{name_model}",
        name_dataset,
        "after",
        f"{n_steps}_steps",
        prompt_after,
        "results.csv",
    )

    print(evaluate_predictions(csv_path=path_before))
    print(evaluate_predictions(csv_path=path_after))

    print(evaluate_consistency_predictions(csv_path_before=path_before, csv_path_after=path_after))