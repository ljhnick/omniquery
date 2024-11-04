from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from legacy_code.main_web import initialize, retrieve
from pipeline import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init', methods=['POST'])
def init():
    data = request.get_json(force=True)
    api_key = data['api_key']
    folder_path = data['folder_path']
    try:
        initialize(api_key=api_key, folder_path=folder_path)
        return jsonify({"status": "Memory initialized"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(force=True)
    text = data['text']
    version = data['version']

    result = retrieve(text, version)

    return jsonify(result[0])

@app.route('/save_result', methods=['POST'])
def save_result():
    data = request.get_json(force=True)

    save_path = 'data/results.json'
    directory = os.path.dirname(save_path)

    if os.path.exists(directory) == False:
        os.makedirs(directory)
    
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    this_result = {}
    this_result['query'] = data['query']
    this_result['omniquery'] = {
        'answer': data['omniAnswer'],
        'explanation': data['omniExplanation'],
        'correctness': data['correctness-ratingContainerOmniQuery'],
        'credibility': data['credibility-ratingContainerOmniQuery']
    }
    this_result['rag'] = {
        'answer': data['ragAnswer'],
        'explanation': data['ragExplanation'],
        'correctness': data['correctness-ratingContainerRAG'],
        'credibility': data['credibility-ratingContainerRAG']
    }

    results.append(this_result)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    return jsonify({"status": "Saved"}), 200


if __name__ == '__main__':
    app.run(debug=True)