import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import json
import os
import time
import io
import base64
import cv2

from flask import Flask, request, jsonify, send_file
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from src.process.memory import Memory
from src.process.preprocess import ProcessMemoryContent
from src.query.query import QueryHandler


app = Flask(__name__)

def initialize_memory(args):
    # input image, for each media in the folder, build memory node from it.
    raw_data_folder = args.raw_data_folder
    processed_folder = args.processed_folder

    memory = Memory(raw_data_folder, processed_folder)

    memory.preprocess()
    memory.augment()

    return memory


def initialize(api_key="", folder_path=""):
    
    parser = get_args_parser()
    global args
    args = parser.parse_args()

    global memory

    if api_key != '':
        os.environ["OPENAI_API_KEY"] = api_key

    if folder_path != '':
        args.raw_data_folder = folder_path
    memory = initialize_memory(args)

    # return jsonify({"status": "Memory initialized"}), 200

def load_memory_media(path, memory: Memory):
    # check if image or video
    ext = os.path.splitext(path)[1].lower()
    memory_id = os.path.basename(path)
    if ext in ['.jpg', '.jpeg', '.png', '.heic']:
        image = Image.open(path)
        img_io = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(img_io, 'JPEG')
        img_io.seek(0)
        img_based64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    else:
        # this is a video
        # pick the first frame
        video = cv2.VideoCapture(path)
        _, image = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)
        img_based64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    raw_data = {}
    for memory_entry in memory.memory_content_processed:
        if memory_entry['filename'] == memory_id:
            raw_data = memory_entry
            break
    
    result = {
        'image': img_based64,
        'raw_data': raw_data
    }
    
    return result

def retrieve(query, version='lite', topk=50):
    if 'memory' not in globals():
        return jsonify({"error": "Memory not initialized"}), 400

    if not query or query == '':
        return jsonify({"error": "No query provided"}), 400
    
    global memory
    
    start_time = time.time()

    query_handler = QueryHandler(memory)

    if version == 'full':
        result = query_handler.query_memory(query)
    else:
        result = query_handler.query_rag(query, topk=topk)

    answer = result['answer']
    explanation = result['explanation']
    memory_ids = result['memory_ids']
    memory_folder = args.raw_data_folder

    all_media = []
    for memory_id in memory_ids:
        memory_path = os.path.join(memory_folder, memory_id)
        media = load_memory_media(memory_path, memory) # {image: base64, raw_data: dict}
        all_media.append(media)


    time_cost = time.time() - start_time
    result_final = {}
    omni = {}
    rag = {}

    omni['answer'] = answer
    omni['explanation'] = explanation
    omni['images'] = all_media
    omni['time_cost'] = round(time_cost, 1)
    
    result_final = {
        'omniquery': omni,
        'rag': rag
    }
    return result_final, 200

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--raw_data_folder", default="data/raw/", type=str)
    parser.add_argument("--processed_folder", default="data/processed/", type=str)

    return parser


if __name__ == "__main__":
    initialize()
    retrieve("Food pictures during stay at Rosewood in Kona")
