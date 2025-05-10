import os
import json
import pandas as pd

def load_dataset(path):
    """
    Load the dataset from a JSON file.
    Args:
        path (str): The path to the JSON file.
    Returns:
        list: A list of dictionaries containing the dataset.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_dataset(dataset, step_distance=1):
    """
    Preprocess the dataset to create pairs of steps with a specified distance between them.
    Args:
        dataset (list): The dataset containing recipes.
        step_distance (int): The distance between steps to create pairs.
    Returns:
        list: A list of dictionaries containing pairs of steps.
    """
    pairs = []

    for recipe in dataset:
        num_steps = len(recipe["steps"])
        for i in range(num_steps):
            if i + step_distance < num_steps:
                step1 = recipe["steps"][i]
                step2 = recipe["steps"][i + step_distance]
                pairs.append({"step1": step1, "step2": step2})

    return pairs

def create_next_result_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    
    # Get all numeric folder names
    existing_folders = [
        name for name in os.listdir(base_dir)
        if name.isdigit() and os.path.isdir(os.path.join(base_dir, name))
    ]
    
    # Find the next available number
    numbers = [int(name) for name in existing_folders]
    next_number = max(numbers, default=0) + 1
    folder_name = f"{next_number:03d}"
    
    result_folder = os.path.join(base_dir, folder_name)
    os.makedirs(result_folder)
    return result_folder

def save_results(results, output_csv_path):
    """
    Save the results to a CSV file.
    Args:
        results (list): The list of results to save.
        output_csv_path (str): The path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=True)
    
    print(f"Results saved to {output_csv_path}")


def save_info(output_txt_path, name_dataset, type_prompt, step_distance, question_prompt, random_order, message_prompt):
    """
    Save the information to a text file.
    Args:
        output_txt_path (str): The path to the output text file.
        name_dataset (str): The name of the dataset.
        type_prompt (str): The type of prompt used.
        step_distance (int): The distance between steps.
        question_prompt (str): The question prompt used.
        random_order (bool): Whether the order of steps is random.
        message_prompt (str): The message prompt used.
    """
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    with open(output_txt_path, 'w') as f:
        f.write(f"Dataset: {name_dataset}\n")
        f.write(f"Type Prompt: {type_prompt}\n")
        f.write(f"Step Distance: {step_distance}\n")
        f.write(f"Question Prompt: {question_prompt}\n")
        f.write(f"Random Order: {random_order}\n")
        f.write(f"Message Prompt: {message_prompt}\n")
    
    print(f"Info saved to {output_txt_path}")
