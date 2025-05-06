

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoProcessor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration, LlavaForConditionalGeneration
import torch
import os
import requests
from PIL import Image, ImageOps
from io import BytesIO
from tqdm import tqdm
import pandas as pd
import json
import re
from utils import calculate_accuracy, operation_guidance_dict
from torch.utils.data import Dataset


def load_instructblip_hf(args):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )


    model_id = args.model#"Salesforce/instructblip-vicuna-7b"

    model = InstructBlipForConditionalGeneration.from_pretrained(model_id,low_cpu_mem_usage=True,quantization_config=quantization_config)
    processor = InstructBlipProcessor.from_pretrained(model_id)
    return model, processor

def caption_image_instructblip_hf(img_path, prompt,model,processor,args):
    if isinstance(img_path, str):
        if img_path.startswith('http') or img_path.startswith('https'):
            response = requests.get(img_path)
            image_original = Image.open(BytesIO(response.content)).convert('RGB')
            image = ImageOps.exif_transpose(image_original)
        else:
            image_original = Image.open(img_path).convert('RGB')
            image = ImageOps.exif_transpose(image_original)
    elif isinstance(img_path, Image.Image):
        # If img_path is already an Image object, use it directly
        image = img_path.convert('RGB')
    else:
        # Handle other cases or raise an exception if needed
        raise ValueError("Invalid img_path type. It should be a string or an Image object.")
    if args.self_define_prompt is not None:
        wrapped_prompt = args.self_define_prompt.replace('YOUR_PROMPT',prompt)
    else:
        wrapped_prompt = f"""<Image>{prompt} A short answer to the question is"""
    inputs = processor(images=image, text=wrapped_prompt, return_tensors="pt").to('cuda', torch.float16)
    output = model.generate(**inputs, max_new_tokens=250)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text[0].strip()

def caption_image_instructblip_hf_batch_inference(imgs,texts,model,processor,args):
    if args.self_define_prompt is not None:
        prompts = [args.self_define_prompt.replace('YOUR_PROMPT',text) for text in texts]
    else:# The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <Image>{text}  Provide your answer as short as possible. ASSISTANT:""" for text in texts]
        prompts = [f"""<Image>{text} A short answer to the question is""" for text in texts]
    inputs = processor(images=imgs,text=prompts, padding=True, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=250)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return [text for text in generated_text]


def load_llava_hf(args):
    """Load LLaVA model and processor from HuggingFace."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_id = args.model
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    return model, processor

def caption_image_llava_hf_batch_inference(imgs, texts, model, processor, args):
    """Batch inference for LLaVA."""
    if args.self_define_prompt is not None:
        prompts = [args.self_define_prompt.replace('YOUR_PROMPT', text) for text in texts]
    else:
        prompts = [
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            "USER: <image>\n{text}\nAnswer the question using a single word or phrase. ASSISTANT:"
            for text in texts
        ]
    prompts = [prompt.replace('<Image>', '<image>') for prompt in prompts]
    inputs = processor(prompts, images=imgs, padding=True, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=250)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return [text.split("ASSISTANT:")[-1] for text in generated_text]

def caption_image_llava_hf(img_path, prompt, model, processor, args):
    """Generate a caption for an image using LLaVA (single image)."""
    if isinstance(img_path, str):
        if img_path.startswith('http') or img_path.startswith('https'):
            response = requests.get(img_path)
            image_original = Image.open(BytesIO(response.content)).convert('RGB')
            image = ImageOps.exif_transpose(image_original)
        else:
            image_original = Image.open(img_path).convert('RGB')
            image = ImageOps.exif_transpose(image_original)
    elif isinstance(img_path, Image.Image):
        image = img_path.convert('RGB')
    else:
        raise ValueError("Invalid img_path type. It should be a string or an Image object.")
    if args.self_define_prompt is not None:
        raw_prompt = args.self_define_prompt.replace('YOUR_PROMPT', prompt)
    else:
        raw_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: <image>\n{prompt}\nAnswer the question using a single word or phrase. ASSISTANT:"
        )
    raw_prompt = raw_prompt.replace('<Image>', '<image>')
    inputs = processor(text=raw_prompt, images=image, padding=True, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=250)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text[0].split("ASSISTANT:")[-1].strip()

def image_parser(image_file):
    """Parse image file string (comma-separated for multiple images)."""
    return image_file.split(',')

def load_image(image_file):
    """Load an image from a file path or URL."""
    if isinstance(image_file, str):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image_original = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image_original = Image.open(image_file).convert('RGB')
    elif isinstance(image_file, Image.Image):
        image_original = image_file.convert('RGB')
    else:
        raise ValueError("Invalid img_path type. It should be a string or an Image object.")
    return ImageOps.exif_transpose(image_original)

def load_images(image_files):
    """Load a list of images from file paths or URLs."""
    return [load_image(image_file) for image_file in image_files]

def load_llava_official(args):
    """Load LLaVA model and processor from the official repo."""
    from llava.model import LlavaLlamaForCausalLM
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
    from llava.model.builder import load_pretrained_model
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    
    disable_torch_init()
    model_path = args.model
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name
    )
    return tokenizer, model, image_processor, context_len, model_name

def caption_image_llava_official(image_file, qs, model, tokenizer, image_processor, model_name, args):
    """Generate a caption for an image using LLaVA official (single image)."""
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            f"[WARNING] the auto inferred conversation mode is {conv_mode}, while `--conv-mode` is {args.conv_mode}, using {args.conv_mode}"
        )
    else:
        args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if args.self_define_prompt is not None:
        prompt = args.self_define_prompt.replace('<image>\nYOUR_PROMPT', qs)
    image_files = image_parser(image_file)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def read_filtered_df(args):
    """Read and filter the DataFrame based on similarity threshold."""
    prompt_mode = 'WITH_Unanswerable' if 'Unanswerable' in args.self_define_prompt else 'WITHOUT_Unanswerable'
    csv_file_path = f'{args.prediction_result_path}/{args.model}/move_camera/Initial_prediction_{prompt_mode}.csv'
    df = pd.read_csv(csv_file_path)
    filtered_df = df[df['initial_response_similarity'] >= args.similarity_threshold]
    return filtered_df

def initial_predict_llava_official(parsed_data, model, image_processor, tokenizer, caption_img_fn, model_name, args):
    """Run initial prediction for LLaVA official and save results."""
    prompt_mode = 'WITH_Unanswerable' if 'Unanswerable' in args.self_define_prompt else 'WITHOUT_Unanswerable'
    csv_file_path = f'{args.prediction_result_path}/{args.model}/move_camera/Initial_prediction_{prompt_mode}.csv'
    directory = os.path.dirname(csv_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    new_row = {
        'Img_path': ' ',
        'Question': ' ',
        'Response': ' ',
        'Answer': ' ',
    }
    columns = list(new_row.keys())
    try:
        result_record = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        result_record = pd.DataFrame(columns=columns)
    img_folder = f'{args.original_img_folder_path}/{args.image_split}/'
    for i in tqdm(range(len(parsed_data))):
        image_filename = list(parsed_data.keys())[i]
        if image_filename in result_record['Img_path'].tolist():
            continue
        question = parsed_data[image_filename]["question"]
        most_common_answer = parsed_data[image_filename]["most_common_answer"]
        img_path = img_folder + image_filename
        generated_text = caption_img_fn(img_path, question, model=model, tokenizer=tokenizer, image_processor=image_processor, model_name=model_name, args=args)
        new_row = {
            'Img_path': image_filename,
            'Question': question,
            'Response': generated_text,
            'Answer': most_common_answer,
            'initial_response_similarity': calculate_accuracy(generated_text, most_common_answer)
        }
        new_df = pd.DataFrame([new_row])
        result_record = pd.concat([result_record, new_df], ignore_index=True)
        result_record.to_csv(csv_file_path, index=False)
    filtered_result = result_record[result_record['initial_response_similarity'] >= args.similarity_threshold]
    filtered_data = {index: parsed_data[index] for index in parsed_data.keys() if index in filtered_result['Img_path'].tolist()}
    return filtered_result, filtered_data

def load_blip2(args):
    """Load BLIP2 model and processor from HuggingFace."""
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    return model, processor

def caption_image_blip2(img_path, prompt, model, processor, args):
    """Generate a caption for an image using BLIP2 (single image)."""
    if isinstance(img_path, str):
        if img_path.startswith('http') or img_path.startswith('https'):
            response = requests.get(img_path)
            image_original = Image.open(BytesIO(response.content)).convert('RGB')
            image = ImageOps.exif_transpose(image_original)
        else:
            image_original = Image.open(img_path).convert('RGB')
            image = ImageOps.exif_transpose(image_original)
    elif isinstance(img_path, Image.Image):
        image = img_path.convert('RGB')
    else:
        raise ValueError("Invalid img_path type. It should be a string or an Image object.")
    raw_prompt = f"Question: {prompt}\nAnswer:"
    inputs = processor(image, text=raw_prompt, return_tensors="pt").to('cuda', torch.float16)
    model.to('cuda')
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def caption_image_blip2_batch_inference(imgs, texts, model, processor, args):
    """Batch inference for BLIP2."""
    prompts = [f"Question: {text}\nAnswer:" for text in texts]
    inputs = processor(imgs, text=prompts, return_tensors="pt").to('cuda', torch.float16)
    model.to('cuda')
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [i.strip() for i in generated_text]

def initial_predict_hf(parsed_data, model, processor, caption_img_fn, args):
    """Run initial prediction for HuggingFace models and save results."""
    prompt_mode = 'WITH_Unanswerable' if 'Unanswerable' in args.self_define_prompt else 'WITHOUT_Unanswerable'
    csv_file_path = f'{args.prediction_result_path}/{args.model}/move_camera/Initial_prediction_{prompt_mode}.csv'
    directory = os.path.dirname(csv_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    new_row = {
        'Img_path': ' ',
        'Question': ' ',
        'Response': ' ',
        'Answer': ' ',
    }
    columns = list(new_row.keys())
    try:
        result_record = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        result_record = pd.DataFrame(columns=columns)
    img_folder = f'{args.original_img_folder_path}/{args.image_split}/'
    for i in tqdm(range(len(parsed_data))):
        image_filename = list(parsed_data.keys())[i]
        if image_filename in result_record['Img_path'].tolist():
            continue
        question = parsed_data[image_filename]["question"]
        most_common_answer = parsed_data[image_filename]["most_common_answer"]
        img_path = img_folder + image_filename
        prompt = f"{question}"
        generated_text = caption_img_fn(img_path, prompt, model, processor, args)
        new_row = {
            'Img_path': image_filename,
            'Question': question,
            'Response': generated_text,
            'Answer': most_common_answer,
            'initial_response_similarity': calculate_accuracy(generated_text, most_common_answer)
        }
        new_df = pd.DataFrame([new_row])
        result_record = pd.concat([result_record, new_df], ignore_index=True)
        result_record.to_csv(csv_file_path, index=False)
    filtered_result = result_record[result_record['initial_response_similarity'] >= args.similarity_threshold]
    filtered_data = {index: parsed_data[index] for index in parsed_data.keys() if index in filtered_result['Img_path'].tolist()}
    return filtered_data

class GuidanceDataset_synthetic(Dataset):
    """Dataset for synthetic guidance data (single perturbation type)."""
    def __init__(self, synthetic_data_info, keyword, args):
        if isinstance(synthetic_data_info, dict):
            data_list = [{'image': i, 'question': synthetic_data_info[i]['question']} for i in synthetic_data_info]
        if isinstance(synthetic_data_info, list):
            data_list = synthetic_data_info
        if isinstance(synthetic_data_info, str):
            with open(synthetic_data_info, 'r') as f:
                data_list = json.load(f)
        self.data_list = [i for i in data_list if i['image'].startswith(f"{keyword.replace(' ','_').upper()}")]
        self.args = args
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        image_path = f'{self.args.target_img_folder_path}/{self.data_list[idx]["image"]}'
        image_original = Image.open(image_path).convert('RGB')
        image = ImageOps.exif_transpose(image_original)
        text = self.data_list[idx]['question']
        return self.data_list[idx]["image"], image, text

class GuidanceDataset_test(Dataset):
    """Dataset for test guidance data."""
    def __init__(self, data_list, img_folder):
        if isinstance(data_list, dict):
            data_list = [{'image': i, 'question': data_list[i]['question']} for i in data_list]
        if isinstance(data_list, list):    
            self.data_list = data_list
        else: 
            raise ValueError('The data_list should be either a list or a dict')
        self.img_folder = img_folder
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        image_path = f'{self.img_folder}/{self.data_list[idx]["image"]}'
        image_original = Image.open(image_path).convert('RGB')
        image = ImageOps.exif_transpose(image_original)
        text = self.data_list[idx]['question']
        return self.data_list[idx]["image"], image, text

def custom_collate(batch):
    """Custom collate function for DataLoader."""
    img_name, images, texts = zip(*batch)
    return img_name, images, texts


