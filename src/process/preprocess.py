# processed_memory
# {'processed_filenames': [], 'processed_memory': {'filename': {}}}
import os
import json

from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

from ..llm.llm import OpenAIWrapper

from utils.exif_utils import read_metadata_from_image, read_metadata_from_video, read_metadata_from_image_exiftool
from utils.video_utils import get_first_frame, transcribe_audio, sample_frames_from_video
from utils.ocr import detect_text

def get_model_info(model_ID, device):
    model = CLIPModel.from_pretrained(model_ID).to(device)
 	# Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    text_model = CLIPTextModel.from_pretrained(model_ID).to(device)
    return model, processor, tokenizer, text_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ID = "openai/clip-vit-base-patch32"
model, processor, tokenizer, text_model = get_model_info(model_ID, device)

class ProcessMemoryContent():
    img_ext_list = ["jpg", "jpeg", "png", "heic"]
    video_ext_list = ["mp4", "mov", "avi"]

    def __init__(
            self,
            raw_data_folder: str,
            processed_folder: str,
            # processed_memory: dict # memory after preprocessing
            ) -> None:
        self.raw_data_folder = raw_data_folder
        self.processed_folder = processed_folder
        # self.processed_memory = processed_memory 

        self.cost = 0

    def load_metadata_and_sort(self):
        files = os.listdir(self.raw_data_folder)

        image_files = {file.split('.')[0] for file in files if file.lower().endswith(tuple(self.img_ext_list))}

        raw_memory_list = []

        print("Loading metadata and sorting ...")
        for file in tqdm(files):
            # print(file)
            if file == ".DS_Store":
                continue

            try:
                filename_no_ext = file.split('.')[0]
                ext = file.split('.')[-1].lower()

                # if '8421' not in filename_no_ext:
                #     continue

                # check if the file has another image file with the same name
                if ext in self.video_ext_list and filename_no_ext in image_files:
                    continue

                filepath = os.path.join(self.raw_data_folder, file)
                if ext in self.img_ext_list:
                    media_type = 'image'
                elif ext in self.video_ext_list:
                    media_type = 'video'
                else:
                    media_type = None
                    continue

                # get metadata first, then order using the timestamp
                if media_type == 'image':
                    metadata = read_metadata_from_image_exiftool(filepath)
                else:
                    metadata = read_metadata_from_video(filepath)

                raw_memory = {
                    'filename': file,
                    'filepath': filepath,
                    'media_type': media_type,
                    'metadata': metadata
                }

                raw_memory_list.append(raw_memory)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # sort the raw memory list by timestamp
        raw_memory_list.sort(key=lambda x: x['metadata']['temporal_info']['date_string'])

        self.raw_memory_with_metadata = raw_memory_list

        
    def is_similar_to_prev_image(self, raw_memory):
        try:
            image = Image.open(raw_memory['filepath'])
        except Exception as e:
            print(f"Error: {e}")
            return False
        
        image_tensor = processor(
            text=None,
            images=image,
            return_tensors="pt",
            padding=True
        )["pixel_values"].to(device)
        embedding = model.get_image_features(image_tensor)
        embedding_as_np = embedding.cpu().detach().numpy()

        if self.prev is None:
            self.prev = {
                'memory': raw_memory,
                'embedding': embedding_as_np
            }
            return

        prev_emb = self.prev['embedding']
        similarity = cosine_similarity(prev_emb, embedding_as_np)

        capture_method = raw_memory['metadata']['capture_method']
        if capture_method == 'photo':
            threshold = 0.85
        else:
            threshold = 0.95

        if similarity[0][0] > threshold:
            this_child = {'filename': raw_memory['filename'], 'filepath': raw_memory['filepath']}
            if 'children' not in self.prev['memory']:
                self.prev['memory']['children'] = []
            self.prev['memory']['children'].append(this_child)
            return True
        else:
            self.prev = {
                'memory': raw_memory,
                'embedding': embedding_as_np
            }
            return False

    def is_similar_to_prev_video(self, raw_memory):
        video_path = raw_memory['filepath']
        first_frame = get_first_frame(video_path)
        # opencv to PIL
        first_frame = Image.fromarray(first_frame)

        frame_tensor = processor(
            text=None,
            images=first_frame,
            return_tensors="pt",
            padding=True
        )["pixel_values"].to(device)
        embedding = model.get_image_features(frame_tensor)
        embedding_as_np = embedding.cpu().detach().numpy()

        if self.prev is None:
            self.prev = {
                'memory': raw_memory,
                'embedding': embedding_as_np
            }
            return
        
        prev_emb = self.prev['embedding']
        similarity = cosine_similarity(prev_emb, embedding_as_np)
        if similarity[0][0] > 0.85:
            this_child = {'filename': raw_memory['filename'], 'filepath': raw_memory['filepath']}
            if 'children' not in self.prev['memory']:
                self.prev['memory']['children'] = []
            self.prev['memory']['children'].append(this_child)
            return True
        else:
            self.prev = {
                'memory': raw_memory,
                'embedding': embedding_as_np
            }
            return False

    def filter_identical_memory(self):

        identical_memory_list = []
        
        print("Filtering identical memory ...")
        for raw_memory in tqdm(self.raw_memory_with_metadata):
            if raw_memory['media_type'] == 'image':
                try:
                    if self.is_similar_to_prev_image(raw_memory):
                        continue
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                identical_memory_list.append(raw_memory)
                
            else:
                if self.is_similar_to_prev_video(raw_memory):
                    continue
                identical_memory_list.append(raw_memory)
        
        # save the identical memory to a file
        self.identical_memory_list = identical_memory_list

        # check if the folder exist
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')
        file_path = 'data/processed/identical_memory_list.json'
        with open(file_path, 'w') as f:
            json.dump(identical_memory_list, f)
        
    def process_identical_memory_content(self):
        for memory in tqdm(self.memory_content_processed[:]):
            if 'content' in memory:
                continue
            media_type = memory['media_type']
            if media_type == 'video':
                path = memory['filepath']
                # only process first 10 seconds of the video
                try:
                    speech = transcribe_audio(path)
                    frames = sample_frames_from_video(path, 4)

                    visual_content, cost = self.llm.generate_visual_content_video(frames, speech)
                except Exception as e:
                    print(f"Error: {e}")
                    self.memory_content_processed.remove(memory)
                    continue

                self.cost += cost

                content = {
                    'caption': visual_content['caption'],
                    'objects': visual_content['objects'],
                    'people': visual_content['people'],
                    'activities': visual_content['activities'],
                    'speech': speech
                }
                memory['content'] = content
            else: # image
                path = memory['filepath']
                image = Image.open(path)
                try:
                    visual_content, cost = self.llm.generate_visual_content(image)
                except Exception as e:
                    print(f"Error: {e}")
                    self.memory_content_processed.remove(memory)
                    continue

                try:
                    text = detect_text(image)
                except Exception as e:
                    raise e

                self.cost += cost
                
                content = {
                    'caption': visual_content['caption'],
                    'objects': visual_content['objects'],
                    'people': visual_content['people'],
                    'activities': visual_content['activities'],
                    'text': text
                }
                memory['content'] = content

    def _save(self, save_data, save_path):
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

    def process(self):

        # check if the identical memory has already been processed
        identical_memory_list_path = os.path.join(self.processed_folder, 'identical_memory_list.json')
        if not os.path.exists(identical_memory_list_path):
            self.load_metadata_and_sort()
            self.prev = None
            self.filter_identical_memory()
        else:
            with open(identical_memory_list_path, 'r') as f:
                self.identical_memory_list = json.load(f)

        image_num = 0
        video_num = 0
        for memory in self.identical_memory_list:
            if memory['media_type'] == 'image':
                image_num += 1
            else:
                video_num += 1
        print(f"Identical memory has been processed. \nImage: {image_num}, Video: {video_num}")

        self.llm = OpenAIWrapper()
        memory_cotent_processed_path = os.path.join(self.processed_folder, 'memory_content_processed.json')
        if not os.path.exists(memory_cotent_processed_path):
            self.memory_content_processed = self.identical_memory_list.copy()
        else:
            with open(memory_cotent_processed_path, 'r') as f:
                self.memory_content_processed = json.load(f)

        try:
            self.process_identical_memory_content()
        except Exception as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("Process interrupted.")
            
        save_path = os.path.join(self.processed_folder, 'memory_content_processed.json')
        self._save(self.memory_content_processed, save_path)