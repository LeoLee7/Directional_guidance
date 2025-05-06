import json
import openai
import base64
import requests
from tqdm import tqdm
import os
import shutil
import numpy as np
import random
import pandas as pd

operation_guidance_dict = {'move to down': 'up',
                            'move to up' : 'down',
                            'move to left': 'right',
                            'move to right': 'left',
                            "{'MOVE_TO_DOWN'}": 'up',
                            "{'MOVE_TO_UP'}" : 'down',
                            "{'MOVE_TO_LEFT'}": 'right',
                            "{'MOVE_TO_RIGHT'}": 'left',}

def get_irrlevant_paired_samples(args):
    """
    Generate irrelevant (unmatched) image-question pairs for negative training samples.
    Uses OpenAI API to rephrase questions if needed. All paths are provided via args.
    """
    json_file_path = f'{args.grounding_annotation_path}/{args.image_split}_grounding.json'
    with open(json_file_path, "r") as json_file:
        parsed_data = json.load(json_file)     
    if not os.path.exists(f'{args.prediction_result_path}/X_cases/unmatched_images_for_each_text.json'):
        print('Generating the irrelevant json files...')
        result_path = f'{args.prediction_result_path}/X_cases/rephrased_question.json'
        if os.path.exists(result_path):
            with open(result_path,'r')as file:
                rephrased_text = json.load(file)
        else:
            rephrased_text = []
            openai.api_key = os.environ.get('OPENAI_API_KEY')
            for key in tqdm(list(parsed_data.keys())):
                question = parsed_data[key]['question']
                answer = parsed_data[key]['most_common_answer']
                s = question + answer
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant to ask a question based on the text."},
                        {"role": "user", "content": f"Please ask a question derived from the following question:{s}. The generated question should include the answer and asking for the location of visual evidence for that answer in an image. Please avoid using 'where','find' or 'visual evidence'"},
                    ],
                    temperature=0.9
                )
                rephrased_text.append(response.choices[0].message.content)
            result_dir = os.path.dirname(result_path)
            os.makedirs(result_dir,exist_ok=True)
            with open(result_path,'w')as file:
                json.dump(rephrased_text,file,indent=4)
        questions = rephrased_text
        img_paths = list(parsed_data.keys())
        unrelated_img_list_2 = []
        for i in range(len(parsed_data)):
            pool = np.arange(len(parsed_data))
            candidate = np.delete(pool,i)
            random.seed(i)
            irrelevant_index = random.sample(list(candidate),2)
            unrelated_img_list_2.append(irrelevant_index)    
        json_list = []
        for i,question in tqdm(enumerate(questions)):
            for j in unrelated_img_list_2[i]:
                id = img_paths[j]
                caption = "none of the other options"
                json_dict = {'id':id,
                             'image':id,
                             'conversations':[
                                 {
                                     'from': 'human',
                                     'value': f'<image>\nQuestion:{question}'
                                 },
                                 {
                                     'from': 'gpt',
                                     'value': f'{caption}'
                                 }
                             ]}
                json_list.append(json_dict)
        with open(f'{args.prediction_result_path}/X_cases/unmatched_images_for_each_text.json','w') as file:
            json.dump(json_list,file,indent=4)
    else:
        print('The irrelevant json files are already generated.')
    print('Now shutil the images to the guidance dataset...')
    src_folder = os.path.join(args.original_img_folder_path,args.image_split)
    target_folder = args.target_img_folder_path
    for img in tqdm(list(parsed_data.keys())):
        src_path  = os.path.join(src_folder, img)
        target_path = os.path.join(target_folder, img)
        if os.path.exists(target_path):
            continue
        shutil.copy(src_path,target_path)
    print('Images shuilted to the guidance dataset.')

def get_shuffled_question(answer, random_number = 42):
    """
    Shuffle answer options and return the new option string and correct answer letter.
    """
    options = ['left', 'right', 'up', 'down', 'none of the other options','leave it unchanged']
    random.seed(random_number)
    random.shuffle(options)
    option_mapping = {}
    letters = ['A', 'B', 'C', 'D', 'E','F']
    for i in range(len(options)):
        option_mapping[letters[i]] = options[i]
    new_option = ''
    for letter, option in option_mapping.items():
        new_option += letter + '.' + option + '\n'    
    correct_answer = answer
    correct_index = options.index(correct_answer)
    index_list = ['A','B','C','D','E','F']
    return new_option, index_list[correct_index]

def shuffle_the_letters(args):
    """
    Shuffle the answer options in the final training JSON.
    """
    print('shuffling the letters...')
    with open(f'{args.prediction_result_path}/{args.model}/three_case_unshuffled.json','r') as f:
        json_list = json.load(f)
    shuffled_json_list = []
    random_seed = 42
    for i in tqdm(json_list):
        id = i['id']
        action = i['conversations'][1]['value']
        question = i['conversations'][0]['value'].split('Question:')[1]
        choices, answer = get_shuffled_question(action,random_seed)
        json_dict = {'id':id,
             'image':id,
             'conversations':[
                 {
                     'from': 'human',
                     'value': f"<image>\nQuestion:{question}\nTo better answer the question, how should the camera be moved?\n{choices}Answer with the option's letter from the given choices directly."
                 },
                 {
                     'from': 'gpt',
                     'value': f'{answer}'
                 }
             ]}
        shuffled_json_list.append(json_dict)
        random_seed += 1
    with open(f'{args.prediction_result_path}/{args.model}/train.json','w') as file:
        json.dump(shuffled_json_list, file, indent=4)
   
def integrate_json_files(args):
    """
    Integrate all generated JSON files into a single training set.
    """
    folder_path = f'{args.prediction_result_path}/{args.model}/move_camera/json_file/'
    json_files = [i for i in os.listdir(folder_path) if (i.startswith('move') and i.endswith('json'))]
    all_json = []
    for file in json_files:
        with open(folder_path+file, "r") as f:
             conv_json = json.load(f)
        all_json.extend(conv_json)
    up = [i for i in all_json if i['conversations'][1]['value'] == 'up']
    down = [i for i in all_json if i['conversations'][1]['value'] == 'down']
    left = [i for i in all_json if i['conversations'][1]['value'] == 'left']
    right = [i for i in all_json if i['conversations'][1]['value'] == 'right']    
    unchanged = [i for i in all_json if i['conversations'][1]['value'] == 'leave it unchanged']
    random_seed = 42
    random.seed(random_seed)
    O_case_num = args.O_case_num if args.O_case_num is not None else int(np.mean([len(up),len(down),len(right),len(left)]))
    if len(unchanged)>O_case_num:
        json_list = random.sample(unchanged, O_case_num)
    else:
        json_list = unchanged        
    for i in [right, left, up, down]:
        json_list.extend(i)
    random.seed(random_seed)
    random.shuffle(json_list)
    file_path = f'{args.prediction_result_path}/{args.model}/move_camera/Move_camera_Able_to_answer.json'
    with open(file_path, 'w') as json_file:
        json.dump(json_list, json_file, indent = 4)
    with open(f'{args.prediction_result_path}/X_cases/unmatched_images_for_each_text.json','r') as f:
        x_case = json.load(f)
    json_list.extend(x_case)
    random.seed(random_seed)
    random.shuffle(json_list)
    with open(f'{args.prediction_result_path}/{args.model}/three_case_unshuffled.json','w') as file:
        json.dump(json_list,file,indent=4)

def from_binary_to_json(args):
    """
    Convert binary similarity results to JSON format for training.
    """
    print('Generating json files...')
    for keyword in args.perturbation:
        csv_folder = f'{args.prediction_result_path}/{args.model}/move_camera/similarity_score'
        csv_file_name = [i for i in os.listdir(csv_folder) if keyword in i][0]
        csv_file_path = os.path.join(csv_folder,csv_file_name)
        df = pd.read_csv(csv_file_path)
        json_file_path = os.path.join(csv_folder.replace('similarity_score','json_file'),keyword + '.json')
        if not os.path.exists(os.path.dirname(json_file_path)):
            os.makedirs(os.path.dirname(json_file_path))
        if not os.path.exists(json_file_path):
            with open(json_file_path, "w") as json_file:
                json.dump([], json_file)
        for img_name in tqdm(df['Img_path'].tolist()):
            gen = get_img_caption(df, img_name, operation_guidance_dict,args)
            with open(json_file_path, "r") as json_file:
                data_list = json.load(json_file)
            batch_size = 10  # Choose an appropriate batch size
            batch_updates = []
            for img_menu, json_dict in gen:
                aug_img_name = json_dict['id']
                if not is_record_present(json_dict, data_list):
                    try:
                        data_list.append(json_dict)
                        batch_updates.append(json_dict)
                    except:
                        continue
                if len(batch_updates) >= batch_size:
                    with open(json_file_path, "w") as json_file:
                        json.dump(data_list, json_file)
                    batch_updates = []
            if batch_updates:
                with open(json_file_path, "w") as json_file:
                    json.dump(data_list, json_file, indent = 4)

def is_record_present(new_data, existing_data):
    """
    Check if a record is already present in the data list.
    """
    return any(new_data['id'] == record.get('id') for record in existing_data)

def get_img_caption(df, img_name,operation_guidance_dict,args):
    """
    Generate image-caption pairs for a given dataframe row.
    """
    row = df[df['Img_path']==img_name].iloc[0]
    perturbation_type = row['Perturbation_type']
    question = row['Question']
    response = row['Answer']
    for param in [f'binary_0.{i}' for i in range(args.perturbation_range[0],args.perturbation_range[1]+1)]:
        status = row[param]
        id = perturbation_type.upper().replace(' ','_') + '_' + param.replace('binary','param') + '_' +img_name
        if status == 'Unchanged':
            caption = 'leave it unchanged'
        elif status == 'Changed':
            caption = operation_guidance_dict[perturbation_type.replace('_',' ').lower()]
        json_dict = {'id':id,
                     'image':id,
                     'conversations':[
                         {
                             'from': 'human',
                             'value': f'<image>\nQuestion:{question}'
                         },
                         {
                             'from': 'gpt',
                             'value': f'{caption}'
                         }
                     ]}
        img_menu = {'original_img_name': img_name,
                           'operation': perturbation_type,
                           'magnitude': param.split('_')[-1],
                           'new_img_name':id}
        yield img_menu,json_dict

def process_results_to_binary(args):
    """
    Process model results to binary similarity format for downstream training data generation.
    """
    print('Inferenced on all synthetic images. Now checking the similarities in binary format...')
    perturbation_control_type = 'customized_ratio'
    for keyword in args.perturbation:
        csv_folder = f'{args.prediction_result_path}/{args.model}/move_camera'
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
        csv_file_name = [i for i in os.listdir(csv_folder) if keyword in i][0]
        csv_file_path = os.path.join(csv_folder,csv_file_name)
        param_list = [f'param_0.{i}' for i in range(args.perturbation_range[0],args.perturbation_range[1]+1)]
        df = pd.read_csv(csv_file_path)
        similarity_name_list = [param.replace('param_','similarity_score_') for param in param_list]
        for similarity_name, param_name in zip(similarity_name_list, param_list):
               df[similarity_name] = df.apply(lambda row: calculate_accuracy(row[param_name], row['Answer']), axis=1)
        df['initial_response_similarity'] = df.apply(lambda row: calculate_accuracy(
            row['Response'],
            row['Answer']
        ), axis=1)
        def process_to_binary(df):
            binary_columns = [f'binary_0.{i}' for i in range(args.perturbation_range[0],args.perturbation_range[1]+1)]
            for col,similarity in zip(binary_columns,similarity_name_list):
                df[col] = df[similarity].apply(lambda x: 'Changed' if x < args.similarity_threshold else 'Unchanged')
            return df
        df_binary = process_to_binary(df)
        target_path = os.path.join(csv_folder,'similarity_score',csv_file_name)
        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path))
        df_binary.to_csv(target_path,index=None)

def calculate_accuracy(response, answers):
    """
    Calculate the accuracy of a response compared to the ground truth answers.
    """
    response = str(response)
    answers = str(answers)
    count = 0
    answer_list = answers.lower().replace(',',' ').replace('/',' ').split(' ')
    response_list = response.lower().replace(',',' ').replace('/',' ').split(' ')
    for answer in answer_list:
        if answer in response_list:
            count = count + 1 
    return count/len(answer_list)