# Directional Guidance

This repository supports the experiments described in our NeurIPS 2024 paper: [Right this way: Can VLMs Guide Us to See More to
Answer Questions?](https://proceedings.neurips.cc/paper_files/paper/2024/file/efe4e50d492fedc0dfd2959f3320a974-Paper-Conference.pdf). It contains code for generating synthetic training datasets for vision-language models (VLMs) through a combination of perturbation-based data augmentation and model inference. It 


The Directional Guidance benchmark dataset is available at 🤗 [Huggingface datacard](https://huggingface.co/datasets/LeoLee7/Directional_Guidance).


## Main Features
- Generate synthetic images with spatial perturbations (move up, down, left, right)
- Run inference with various VLMs (LLaVA, BLIP2, InstructBLIP, etc.)
- Save results and prepare data for downstream training

## Code Structure and Workflow

### High-Level Workflow
1. **Data Preparation:**
   - The code loads image annotations from [VizWiz Grounding](https://vizwiz.org/tasks-and-datasets/answer-grounding-for-vqa/) and prepares the dataset for augmentation and inference.
2. **Synthetic Image Generation (Model-Agnostic):**
   - For each image, spatial perturbations (move up, down, left, right) are applied to create synthetic variants using segmentation masks
   - The synthetic images can be generated once and stored, making them available for all supported VLMs
   - Perturbation process:
     1. Load original image and corresponding segmentation mask
     2. Apply spatial perturbations by cropping and moving the segmented object within a configurable range.
     3. Save perturbed images to target directory (specified by `--target_img_folder_path`)
   - This model-agnostic approach allows easy addition of new models without regenerating synthetic data.

3. **Model Inference (Model-in-the-Loop):**
   - Visual-language models (VLMs) are used to generate captions or answers for both original and synthetic images. Multiple VLMs are supported (LLaVA, BLIP2, InstructBLIP, etc.).
4. **Result Processing:**
   - The results are processed to compute similarity scores, filter data, and convert outputs into training-ready JSON files.
5. **Negative and Irrelevant Sample Generation:**
   - The pipeline generates negative and irrelevant samples and shuffles answer options to create robust training data.
6. **Integration:**
   - The pipeline combines all generated data (original images, synthetic variants, and negative samples) into a single JSON file and randomly shuffles it to create the final training dataset. The training data format follows LLaVA's finetuning data doc: [reference](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md).


## Folder Structure
```
.
├── generate_train_dataset_main.py  # Main entry point
├── vlm_inference.py                # Model inference utilities
├── utils.py                        # Utility functions
├── data_augmentation/
│   ├── __init__.py
│   └── functions.py                # Image augmentation functions
```
### File/Module Overview
- **generate_train_dataset_main.py**: The main entry point. Handles argument parsing, orchestrates the workflow (data loading, augmentation, inference, result processing, and integration). Calls functions from all other modules.
- **vlm_inference.py**: Provides model loading and inference utilities for various VLMs (LLaVA, BLIP2, InstructBLIP, etc.). Contains functions for both single and batch inference, as well as dataset classes for synthetic/test data.
- **utils.py**: Contains utility functions for post-processing, such as generating negative samples, shuffling answer options, integrating JSON files, converting similarity results to training data, and accuracy calculation.
- **data_augmentation/functions.py**: Implements image augmentation and perturbation functions used to generate synthetic images from originals and masks.

The code is modular: you can swap in new models, change augmentation strategies, or adapt the pipeline for new datasets by modifying the relevant module.

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   - Download and prepare your dataset:
     - For [VizWiz Grounding dataset](https://vizwiz.org/tasks-and-datasets/answer-grounding-for-vqa/):
       ```bash
       # Download VizWiz Grounding annotations and masks
       wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip
       ```
     - You can use any other dataset with grounding information by:
       1. Preparing images and grounding annotations in a similar structure
       2. Modifying the data loading code accordingly
   - Place the downloaded files in appropriate folders and specify paths via arguments:
     ```bash
       --original_img_folder_path /path/to/vizwiz \
       --grounding_annotation_path /path/to/vizwiz_grounding_dataset/annotation \
       --grounding_mask_png_path /path/to/vizwiz_grounding_dataset/grounding_area/val
     ```

3. **Set environment variables (if using OpenAI):**
   - If you use any functionality that calls the OpenAI API, set your API key. For generating irrelevant (X) cases with OpenAI API:
     ```bash
     export OPENAI_API_KEY=your_openai_key_here
     ```
   - Required for `get_irrlevant_paired_samples()` in `utils.py`.


4. **Run the main script**:

   ```bash
   # Example script: generate_llava_7b_hf.sh
   python generate_train_dataset_main.py \
      --data_root /pvcvolume \
      --model llava-hf/llava-1.5-7b-hf  \  # Specify which VLM model to use
      --inference \   # Run model inference on images
      --batch_inference \   # Enable batch processing for faster inference for Huggingface models
      --generate_synthetic_images \  # Generate the cropped images if it has not been done previously
      --save_synthetic_images \  # Save the cropped images if it has not been done previously
      --perturbation_range 1 9 \  # Customize the perturbation range (from 1 to 9)
      --self_define_prompt "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <Image>\nYOUR_PROMPT \nAnswer the question using a single word or phrase. ASSISTANT:" \
   ```



   This script:
   - Uses InstructBLIP-7B model
   - Enables batch inference for faster processing
   - Generates and saves synthetic images (This step can be skipped by setting `generate_synthetic_images` and `save_synthetic_images` **False** if synthetic images already exist)
   - Sets perturbation range from 1-9
   - Uses a custom prompt template without unanswerable option

   Key arguments:
   - `--model`: Which vision-language model to use (e.g. InstructBLIP, LLaVA, BLIP2)
   - `--batch_inference`: Process multiple images at once for faster inference
   - `--data_root`: Base directory containing dataset and output folders
   - `--inference`: Flag to run model inference on the dataset

   See `generate_train_dataset_main.py` for all available arguments and their descriptions.



## Requirements
- Python 3.8+
- torch
- transformers
- opencv-python
- numpy
- pandas
- tqdm
- Pillow

See `requirements.txt` for details.

## Notes
- **All file and folder paths are configurable via command-line arguments.**
- Please see the docstrings in the code for further details on function usage.


### Common Issues & Solutions
- Model Loading:
  - Check huggingface_model_path is set correctly
  - Verify model weights are downloaded
  - Make sure the model name is correctly specified

- Missing Files:
  - Ensure all paths in args are valid
  - Check prediction_result_path for intermediate outputs

- Image Token Formats:
  - Each model requires specific image token formats:
    - LLaVA: Uses "<**I**mage>" (uppercase I)
    - InstructBLIP: Uses "<**i**mage>" (lowercase i)
  - The customized format is defined in self_define_prompt argument
  - If you see token broadcasting errors:
    - Double check you're using the correct format for your model
    - Review the prompt templates in `vlm_inference.py`


## Citation

If you find this work helpful, we will appreciate it if you cite:

```bibtex
@inproceedings{liu2024right,
 title = {Right this way: Can VLMs Guide Us to See More to Answer Questions?},
 author = {Liu, Li and Yang, Diji and Zhong, Sijia and Tholeti, Kalyana Suma Sree and Ding, Lei and Zhang, Yi and Gilpin, Leilani H.},
 booktitle = {Advances in Neural Information Processing Systems},
 year = {2024}
}

```

## License
MIT 
