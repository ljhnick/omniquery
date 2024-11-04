# OmniQuery: Contextually Augmenting Captured Multimodal Memories to Enable Personal Question Answering

## What is OmniQuery
OmniQuery enables **free-form question answering** on massive personal photo album data (photos and videos). It leverages RAG and applies **Contextual Augmentation** to the indexing process to enhance semantic retrieval given any queries. For more details and examples, please refer to our [paper](https://arxiv.org/abs/2409.08250) and [project page](https://jiahaoli.net/omniquery/).


## To-Do List

- [x] Release the source code of OmniQuery.
- [ ] Add parameter control (e.g., topK).
- [ ] Enable support for open-source language models (LLMs).


## Installation
### Prerequisites

Make sure you have [Homebrew](https://brew.sh/) and [Anaconda](https://www.anaconda.com/download) installed in your machine.

Clone this repo:
```
git clone https://github.com/ljhnick/omniquery.git
```

Install `exiftool` and `ffmpeg` globally.
```
brew install exiftool
brew install ffmpeg
```
Create a conda environment and upgrade `pip` and `wheel`.
```
conda create --name omniquery python=3.10.14 -y
conda activate omniquery
pip install --upgrade pip wheel
```
Install dependencies
```
pip install -r requirements.txt
```

### Set up API keys
OmniQuery currently uses Google's cloud vision API for OCR and openAI's API family for captioning, reasoning, question answering, etc.
Thus before running OmniQuery on your own images, you need to set up the API keys first.

For Google cloud vision API, plese follow the [link](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-user-cred) to generate a local credential (`.json` file). 

Create a `.env` file in the root folder and add your API keys:
```
OPENAI_API_KEY="your_api_key"
GOOGLE_APPLICATION_CREDENTIALS="<path_to_credential_json>"
```

## Running OmniQuery
1. **Prepare Your Data**: Download the **photos** and **videos** from your iOS device to the `<root>/data/raw/` folder (we recommend using [ImageCapture](https://support.apple.com/guide/image-capture/imgcp1003/mac) to transfer photos and videos if stored locally on your phone).

2. Then index your memory with contextual augmentation (will take a while):
```
python init.py
```

3. Once the indexing is finished, start the web app:
```
python frontend/app.py
```

4. Navigate to [localhost:5000](http://127.0.0.1:5000). You should see OmniQuery running in your browser. 

5. Every time you run OmniQuery, you should navigate to Settings and Initialize the app first (the button will be grayed out once you successfully initialized the app). 

6. Additionally, you can toggle between `OmniQuery (Full)` and `OmniQuery Lite`. `OmniQuery Lite` is faster and cheaper but it does not perform query augmentation. We suggest using `OmniQuery Lite` first to try out some questions.

## Credits
OmniQuery is created by [Jiahao Nick Li](https://jiahaoli.net/), [Zhuohao Zhang](https://zhuohaozhang.com/), and [Jiaju Ma](https://majiaju.io/).

## Citation
```bibtex
@misc{li2024omniquerycontextuallyaugmentingcaptured,
    title={OmniQuery: Contextually Augmenting Captured Multimodal Memory to Enable Personal Question Answering}, 
    author={Jiahao Nick Li and Zhuohao Jerry Zhang and Jiaju Ma},
    year={2024},
    eprint={2409.08250},
    archivePrefix={arXiv},
    primaryClass={cs.HC},
    url={https://arxiv.org/abs/2409.08250}, 
}
```


## License

The software is available under the [MIT License](https://github.com/poloclub/wizmap/blob/master/LICENSE).

## Contact

If you have any questions, feel free to [open an issue](https://github.com/ljhnick/omniquery/issues/new) or contact [Jiahao Nick Li](https://jiahaoli.net/).