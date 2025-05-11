import os
import torch
import random
import warnings
from llava.model.builder import load_pretrained_model

from function import eval_qwen, eval_llava
from utils import load_dataset, preprocess_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from config import *

if __name__ == "__main__":
    torch.manual_seed(1234)
    random.seed(1234)

    dataset = load_dataset(PATH_DATASET)
    name_dataset = os.path.basename(PATH_DATASET).split(".")[0]

    dataset = preprocess_dataset(dataset, step_distance=STEP_DISTANCE)

    print(len(dataset))

    if MODEL == "qwen":

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map=DEVICE, cache_dir=PATH_MODELS, attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=PATH_MODELS, device_map=DEVICE)

        eval_qwen(
            model=model,
            processor=processor,
            dataset=dataset,
            name_dataset=name_dataset,
            question_prompt=PROMPT,
            type_prompt=TYPE_PROMPT,
            random_order=RANDOM_ORDER,
        )

    elif MODEL == "llava":

        warnings.filterwarnings("ignore")
        llava_model_args = {
        "multimodal": True,
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config

        tokenizer, model, image_processor, max_length = load_pretrained_model("lmms-lab/llava-onevision-qwen2-7b-ov", None, "llava_qwen", device_map=DEVICE, cache_dir=PATH_MODELS, **llava_model_args)


        eval_llava(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            max_length=max_length,
            dataset=dataset,
            name_dataset=name_dataset,
            question_prompt=PROMPT,
            type_prompt=TYPE_PROMPT,
            random_order=RANDOM_ORDER,
        )

    else:

        print("Invalid model name")    