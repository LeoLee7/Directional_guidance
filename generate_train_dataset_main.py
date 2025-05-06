import argparse
import json
import os
import sys
from typing import List, Dict, Any

import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    AutoProcessor
)

from data_augmentation.functions import offset
from vlm_inference import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

def main(args):
    json_file_path = os.path.join(args.grounding_annotation_path, f'{args.image_split}_grounding.json')
    
    # Load dataset annotations
    with open(json_file_path, "r") as json_file:
        parsed_data = json.load(json_file)

    if args.inference: # if not os.path.exists(os.path.join(args.prediction_result_path, args.model, 'move_camera', 'json_file')):
        if args.model.startswith('liuhaotian/llava-v1.5'): # Frome LLaVA official repo: https://github.com/haotian-liu/LLaVA
            from llava.model import LlavaLlamaForCausalLM
            handle_llava_official_inference(args, parsed_data)
        elif args.model.startswith('llava-hf/llava-1.5'): # Frome LLaVA HuggingFace repo: https://huggingface.co/llava-hf/llava-1.5
            handle_llava_hf_inference(args, parsed_data)
        elif args.model == 'blip2-opt-2.7b': # Frome BLIP2 HuggingFace repo: https://huggingface.co/Salesforce/blip2-opt-2.7b
            handle_blip2_inference(args, parsed_data)
        elif 'Salesforce/instructblip-vicuna' in args.model: # Frome InstructBlip HuggingFace repo: https://huggingface.co/Salesforce/instructblip-vicuna-7b
            handle_instructblip_inference(args, parsed_data)
    print(args.model)

    # Process results and generate training data
    process_results_to_binary(args)
    from_binary_to_json(args)
    get_irrlevant_paired_samples(args)
    integrate_json_files(args)
    shuffle_the_letters(args)

def handle_llava_official_inference(args, parsed_data):
    if args.batch_inference:
        pass
    else:
        print(f'Single inference for {args.model}')
        tokenizer, model, image_processor, context_len, model_name = load_llava_official(args)
        caption_img_fn = caption_image_llava_official

        prompt_mode = 'WITH_Unanswerable' if 'Unanswerable' in args.self_define_prompt else 'WITHOUT_Unanswerable'
        csv_file_path = os.path.join(args.prediction_result_path, args.model, 'move_camera', f'Initial_prediction_{prompt_mode}.csv')
        
        if os.path.exists(csv_file_path):
            filtered_df = read_filtered_df(args)
        else:
            filtered_df, filtered_data = initial_predict_llava_official(parsed_data, model, image_processor, tokenizer, caption_img_fn, model_name, args)

        inference_on_synthetic_images_and_save_to_csv(args, filtered_df, caption_img_fn, model, tokenizer, image_processor, model_name)

def handle_llava_hf_inference(args, parsed_data):
    model, processor = load_llava_hf(args)
    if args.batch_inference:
        print(f'Batch inference for {args.model}')
        caption_img_fn = caption_image_llava_hf
        filtered_data = initial_predict_hf(parsed_data, model, processor, caption_img_fn, args)
        aug_img_list = generate_synthetic_images(args, filtered_data)
        caption_img_fn = caption_image_llava_hf_batch_inference
        batch_inference_on_synthetic_images_and_save_to_csv(args, aug_img_list, parsed_data, model, processor, caption_img_fn)

def handle_blip2_inference(args, parsed_data):
    model, processor = load_blip2(args)
    if args.batch_inference:
        print('Batch inference for blip2-opt-2.7b')
        caption_img_fn = caption_image_blip2
        filtered_data = initial_predict_hf(parsed_data, model, processor, caption_img_fn, args)
        aug_img_list = generate_synthetic_images(args, filtered_data)
        caption_img_fn = caption_image_blip2_batch_inference
        batch_inference_on_synthetic_images_and_save_to_csv(args, aug_img_list, parsed_data, model, processor, caption_img_fn)

def handle_instructblip_inference(args, parsed_data):
    model, processor = load_instructblip_hf(args)
    if args.batch_inference:
        print(f'Batch inference for {args.model}')
        caption_img_fn = caption_image_instructblip_hf
        filtered_data = initial_predict_hf(parsed_data, model, processor, caption_img_fn, args)
        aug_img_list = generate_synthetic_images(args, filtered_data)
        caption_img_fn = caption_image_instructblip_hf_batch_inference
        batch_inference_on_synthetic_images_and_save_to_csv(args, aug_img_list, parsed_data, model, processor, caption_img_fn)

def generate_synthetic_images(args, target_data): 
    synthetic_data_info_path = os.path.join(args.prediction_result_path, args.model, 'move_camera', 'synthetic_data', 'img_names_and_questions.json')

    if args.generate_synthetic_images or not os.path.exists(synthetic_data_info_path):
        print('Generating and saving the synthetic images')
        img_folder = os.path.join(args.original_img_folder_path, args.image_split)
        mask_folder = args.grounding_mask_png_path
        folder_name = args.target_img_folder_path
        os.makedirs(folder_name, exist_ok=True)

        aug_img_list = []
        perturbation_types = ['move to up', 'move to down', 'move to left', 'move to right']

        for perturbation_type in perturbation_types:
            for img_name in tqdm(target_data.keys()):
                for param in [f'param_0.{i}' for i in range(args.perturbation_range[0], args.perturbation_range[1] + 1)]:
                    aug_img_name = f"{perturbation_type.upper().replace(' ', '_')}_{param}_{img_name}"
                    magnitude_str = float(param.split('_')[1])
                    
                    if args.save_synthetic_images and not os.path.exists(os.path.join(folder_name, aug_img_name)):
                        try:
                            aug_img = generate_img(img_name, perturbation_type, magnitude_str, img_folder=img_folder, mask_folder=mask_folder)
                            cv2.imwrite(os.path.join(folder_name, aug_img_name), aug_img)
                        except Exception as e:
                            print(f"Error generating {aug_img_name}: {str(e)}")
                            continue
                    
                    aug_img_list.append({
                        'image': aug_img_name,
                        'question': target_data[img_name]['question']
                    })

        os.makedirs(os.path.dirname(synthetic_data_info_path), exist_ok=True)
        with open(synthetic_data_info_path, 'w') as f:
            json.dump(aug_img_list, f, indent=4)
    else:
        with open(synthetic_data_info_path, 'r') as f:
            aug_img_list = json.load(f)
    
    return aug_img_list

def generate_img(img_name, perturbation_type, magnitude_str, img_folder, mask_folder):
    if perturbation_type in ['move to down', 'move to up', 'move to left', 'move to right']:
        magnitude = float(magnitude_str)
        return offset(img_name, offset_param=magnitude, keyword=perturbation_type, crop_mode='cut', 
                     img_folder=img_folder, mask_folder=mask_folder)
    return None

def inference_on_synthetic_images_and_save_to_csv(args, filtered_df, caption_img_fn, model, tokenizer, image_processor, model_name):
    """
    Run inference on synthetic images and save results to CSV for each perturbation type.
    """
    print('Inferencing the synthetic images...')
    perturbation_control_type = 'customized_ratio'
    prompt_mode = 'WITH_Unanswerable' if 'Unanswerable' in args.self_define_prompt else 'WITHOUT_Unanswerable'
    img_folder = os.path.join(args.original_img_folder_path, args.image_split)
    mask_folder = args.grounding_mask_png_path
    folder_name = args.target_img_folder_path

    for keyword in args.perturbation:
        keys_from_range = [f'param_0.{i}' for i in range(args.perturbation_range[0], args.perturbation_range[1] + 1)]
        csv_file_path = os.path.join(args.prediction_result_path, args.model, 'move_camera', f'{keyword}_operation_{prompt_mode}_with_{perturbation_control_type}.csv')
        # Create new columns for each key in keys_from_range
        if not os.path.exists(csv_file_path):
            result_data = filtered_df.copy()
            for key in keys_from_range:
                result_data[key] = np.nan
        else:
            result_data = pd.read_csv(csv_file_path)
        result_data['Perturbation_type'] = keyword

        for index, row in tqdm(result_data.iterrows(), total=result_data.shape[0]):
            image_filename = row['Img_path']
            if not row.isnull().values.any():
                continue
            img_path = os.path.join(img_folder, image_filename)
            prompt = row['Question']
            response_list = []

            for i in range(args.perturbation_range[0], args.perturbation_range[1] + 1):
                if i > 0:
                    try:
                        aug_img = os.path.join(folder_name, f"{keyword.upper().replace(' ', '_')}_param_0.{i}_{image_filename}")
                        generated_text = caption_img_fn(aug_img, prompt, model=model, tokenizer=tokenizer, image_processor=image_processor, model_name=model_name, args=args)
                    except Exception as e:
                        generated_text = np.nan
                        print(f"Error processing {img_path}: {e}")
                    response_list.append(generated_text)

            for key, response in zip(keys_from_range, response_list):
                result_data.at[index, key] = response

            result_data.to_csv(csv_file_path, index=False)

def batch_inference_on_synthetic_images_and_save_to_csv(args, synthetic_data_info, parsed_data, model, processor, caption_img_fn):
    """
    Run batch inference on synthetic images and save results to CSV for each perturbation type.
    """
    from torch.utils.data import DataLoader  # Ensure DataLoader is imported
    for keyword in args.perturbation:
        guidance_synthetic_data = GuidanceDataset_synthetic(synthetic_data_info, keyword, args)
        guidance_synthetic_data_loader = DataLoader(guidance_synthetic_data, batch_size=args.perturbation_range[1] - args.perturbation_range[0] + 1, collate_fn=custom_collate)

        print(f'Batch inference for {args.model}')
        perturbation_control_type = 'customized_ratio'
        prompt_mode = 'WITH_Unanswerable' if 'Unanswerable' in args.self_define_prompt else 'WITHOUT_Unanswerable'
        csv_file_path = os.path.join(args.prediction_result_path, args.model, 'move_camera', f'{keyword}_{prompt_mode}_with_{perturbation_control_type}.csv')
        directory = os.path.dirname(csv_file_path)
        os.makedirs(directory, exist_ok=True)
        responses = [f'response{i}' for i in range(args.perturbation_range[0], args.perturbation_range[1] + 1)]
        keys_from_range = [f'param_0.{i}' for i in range(args.perturbation_range[0], args.perturbation_range[1] + 1)]
        columns = ['Img_path', 'Question', 'Response', 'Answer', 'Perturbation_type'] + keys_from_range
        try:
            result_record = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            result_record = pd.DataFrame(columns=columns)

        for img_names, imgs, texts in tqdm(guidance_synthetic_data_loader):
            img_file_names = set([name[name.index("VizWiz"): ] for name in img_names])
            perturbation_types = set([name.split('_param')[0] for name in img_names])

            assert len(perturbation_types) == 1, f"Error: the perturbation types are not the same: {perturbation_types}"
            assert len(img_file_names) == 1, f"Error: the image file names are not the same: {img_file_names}"
            img_filename = list(img_file_names)[0]
            perturbation_type = list(perturbation_types)[0]

            if img_filename in result_record['Img_path'].tolist():
                continue
            most_common_answer = parsed_data[img_filename]["most_common_answer"]
            question = parsed_data[img_filename]["question"]

            batch_answers = caption_img_fn(imgs, texts, model, processor, args)

            new_row = {
                'Img_path': img_filename,
                'Question': question,
                'Response': 'Place holder',  # temporary Place holder
                'Answer': most_common_answer,
                'Perturbation_type': perturbation_type,
            }
            for key, response in zip(keys_from_range, batch_answers):
                new_row[key] = response

            new_df = pd.DataFrame([new_row])
            result_record = pd.concat([result_record, new_df], ignore_index=True)
            result_record.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training set for guidance")
    
    # File paths
    parser.add_argument('--data_root', type=str, default='/pvcvolume',
                        help='Root directory for all data')

    parser.add_argument('--original_img_folder_path', type=str, default=None)
    parser.add_argument('--grounding_annotation_path', type=str, default=None)
    parser.add_argument('--target_img_folder_path', type=str, default=None)
    parser.add_argument('--prediction_result_path', type=str, default=None)
    parser.add_argument('--grounding_mask_png_path', type=str, default=None)
    parser.add_argument('--huggingface_model_path', type=str, default=None)
    
    # Task settings
    parser.add_argument('--perturbation', type=str, nargs='+',
                       default=['move to up', 'move to down', 'move to left', 'move to right'])
    parser.add_argument('--perturbation_range', nargs='+', type=int, default=[1, 9])
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    parser.add_argument('--generate_synthetic_images', action='store_true')
    parser.add_argument('--save_synthetic_images', action='store_true')
    parser.add_argument('--image_split', type=str, default='val')
    
    # Model settings
    parser.add_argument('--model', type=str, default='llava-hf/llava-1.5')
    parser.add_argument('--batch_inference', action='store_true')
    parser.add_argument('--conv_mode', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--self_define_prompt', type=str,
                       default="""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <Image>\nYOUR_PROMPT \nAnswer the question using a single word or phrase. ASSISTANT:""")
    parser.add_argument('--prompt_mode', type=str, default='WITH_Unanswerable')
    parser.add_argument('--inference', action='store_true')
    
    # Miscellaneous
    parser.add_argument('--O_case_num', type=int, default=None)
    parser.add_argument('--f', type=str, help='Additional configuration file')
    
    args = parser.parse_args()

    if args.original_img_folder_path is None:
        args.original_img_folder_path = os.path.join(args.data_root, 'vizwiz')
    if args.grounding_annotation_path is None:
        args.grounding_annotation_path = os.path.join(args.data_root, 'vizwiz_grounding_dataset', 'annotation')
    if args.target_img_folder_path is None:
        args.target_img_folder_path = os.path.join(args.data_root, 'Guidance_dataset_model_agnostic')
    if args.prediction_result_path is None:
        args.prediction_result_path = os.path.join(args.data_root, 'Guidance', 'Generate_train_set','results')
    if args.grounding_mask_png_path is None:
        args.grounding_mask_png_path = os.path.join(args.data_root, 'vizwiz_grounding_dataset', 'grounding_area', 'val')
    if args.huggingface_model_path is None:
        args.huggingface_model_path = os.path.join(args.data_root, 'models')


    main(args)
