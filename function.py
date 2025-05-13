import base64
from io import BytesIO
import random
import copy
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from config import DEVICE

from utils import save_results, create_next_result_folder, save_info

def eval_qwen(model, processor, dataset, name_dataset, question_prompt, type_prompt, random_order=False):

    if type_prompt not in ["before", "after"]:
        print("Invalid type_prompt")
    
    message_prompt = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This is the first Image of a recipe."},
            {"type": "image", "image": ""},
            {"type": "text", "text": "This is the second Image of a recipe."},
            {"type": "image", "image": ""},
            {"type": "text", "text": ""},
        ],
    }
    ]

    pbar = tqdm(dataset, desc=f"Evaluation...", unit="couples")

    value1 = "step1"
    value2 = "step2"

    results = []

    for couple in pbar:
        if random_order == True:
            value1 = random.choice(["step1", "step2"])

            if value1 == "step1":
                value2 = "step2"
            else:
                value2 = "step1"

        message_prompt[0]["content"][1]["image"] = f"""data:image;base64,{couple[value1]["image"]}"""
        message_prompt[0]["content"][3]["image"] = f"""data:image;base64,{couple[value2]["image"]}"""
        message_prompt[0]["content"][4]["text"] = question_prompt

        texts = [
            processor.apply_chat_template(message_prompt, tokenize=False, add_generation_prompt=True)
        ]
        image_inputs, video_inputs = process_vision_info(message_prompt)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs.to(DEVICE)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output_texts[0] = output_texts[0].strip().lower()
        output_texts[0] = output_texts[0].rstrip(".!?")

        if output_texts[0] not in ["yes", "no"]:
            print(f"Invalid output: {output_texts[0]}")
            exit(1)
        else:
            if type_prompt == "before":
                gold = "yes" if value1 == "step1" else "no"
            elif type_prompt == "after":
                gold = "yes" if value1 == "step2" else "no"

            results.append({
                "step1": couple[value1]["id"],
                "step2": couple[value2]["id"],
                "prediction": output_texts[0],
                "gold": gold,
            })
    
    message_prompt[0]["content"][1]["image"] = ""
    message_prompt[0]["content"][3]["image"] = ""

    result_folder = create_next_result_folder(f"results_qwen/{name_dataset}/{type_prompt}/{dataset[0]['step2']['id']-dataset[0]['step1']['id']}_steps/")

    save_results(results, f"{result_folder}/results.csv")
    save_info(f"{result_folder}/info.txt" ,name_dataset, type_prompt, dataset[0]['step2']['id']-dataset[0]['step1']['id'], question_prompt, random_order, message_prompt)

def eval_llava(tokenizer, model, image_processor, max_length, dataset, name_dataset, question_prompt, type_prompt, random_order=False):
    model.eval()
    model.to(DEVICE)
    if type_prompt not in ["before", "after"]:
        print("Invalid type_prompt")

    pbar = tqdm(dataset, desc=f"Evaluation...", unit="couples")

    value1 = "step1"
    value2 = "step2"

    conv_template = "qwen_1_5"

    results = []

    for couple in pbar:
        if random_order == True:
            value1 = random.choice(["step1", "step2"])

            if value1 == "step1":
                value2 = "step2"
            else:
                value2 = "step1"
        
        img1 = base64.b64decode(couple[value1]["image"])
        img2 = base64.b64decode(couple[value2]["image"])
        img1 = Image.open(BytesIO(img1))
        img2 = Image.open(BytesIO(img2))

        image_tensor = process_images([img1, img2], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=DEVICE) for _image in image_tensor]

        question = "First Image: " + DEFAULT_IMAGE_TOKEN + "Second Image: " + DEFAULT_IMAGE_TOKEN + question_prompt

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
        image_sizes = [img1.size, img2.size]

        cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=128,
        )

        output_texts = tokenizer.batch_decode(cont, skip_special_tokens=True)

        output_texts[0] = output_texts[0].strip().lower()
        output_texts[0] = output_texts[0].rstrip(".!?")

        if output_texts[0] not in ["yes", "no"]:
            print(f"Invalid output: {output_texts[0]}")
            exit(1)
        else:
            if type_prompt == "before":
                gold = "yes" if value1 == "step1" else "no"
            elif type_prompt == "after":
                gold = "yes" if value1 == "step2" else "no"

            results.append({
                "step1": couple[value1]["id"],
                "step2": couple[value2]["id"],
                "prediction": output_texts[0],
                "gold": gold,
            })

    result_folder = create_next_result_folder(f"results_llava/{name_dataset}/{type_prompt}/{dataset[0]['step2']['id']-dataset[0]['step1']['id']}_steps/")

    save_results(results, f"{result_folder}/results.csv")
    save_info(f"{result_folder}/info.txt" ,name_dataset, type_prompt, dataset[0]['step2']['id']-dataset[0]['step1']['id'], question_prompt, random_order, question)

def evaluate_predictions(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Extract prediction and gold columns
    y_pred = df['prediction'].map({'yes': 1, 'no': 0})
    y_true = df['gold'].map({'yes': 1, 'no': 0})
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def evaluate_consistency_predictions(csv_path_before, csv_path_after):
    # Load the CSV
    df_before = pd.read_csv(csv_path_before)
    df_after = pd.read_csv(csv_path_after)

    # Extract prediction and gold columns
    y_pred_before = df_before['prediction'].map({'yes': 1, 'no': 0})
    y_true_before = df_before['gold'].map({'yes': 1, 'no': 0})

    y_pred_after = df_after['prediction'].map({'yes': 1, 'no': 0})
    y_true_after = df_after['gold'].map({'yes': 1, 'no': 0})
    
    assert len(y_pred_before) == len(y_pred_after), "The two datasets must have the same number of samples"
    
    # Compute metrics
    consistency_rate = 0
    correct_consistency_rate = 0
      
    for i in range(len(y_pred_before)):
        if y_true_before[i] == y_true_after[i]:
            print(f"{i} samples have the same gold label")
        elif y_true_before[i] != y_true_after[i]:
            consistency_rate += 1 if y_pred_before[i] != y_pred_after[i] else 0
            correct_consistency_rate += 1 if y_pred_before[i] == y_true_before[i] and y_pred_after[i] == y_true_after[i] else 0
        
    consistency_rate = consistency_rate / len(y_pred_before)
    correct_consistency_rate = correct_consistency_rate / len(y_pred_before)

    return {
        'consistency_rate': consistency_rate,
        'correct_consistency_rate': correct_consistency_rate
    }
            