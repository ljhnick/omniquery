import os
from google.cloud import vision 
import io

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'api_key/textintheworld-6fdf6798885d.json'

def detect_text(image):
    client = vision.ImageAnnotatorClient()
    
    # with open(path, 'rb') as image_file:
    #     content = image_file.read()

    img_byte_arr = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    image = vision.Image(content=img_byte_arr)
    
    # image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if len(texts) == 0:
        return ""
    text_all = texts[0].description
    return text_all
    # print("Texts:")

    # for text in texts:
    #     print(f'\n"{text.description}"')

    #     vertices = [
    #         f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
    #     ]

    #     print("bounds: {}".format(",".join(vertices)))

    # if response.error.message:
    #     raise Exception(
    #         "{}\nFor more info on error messages, check: "
    #         "https://cloud.google.com/apis/design/errors".format(response.error.message)
    #     )
    

# path = "data/raw/version_1_nick/IMG_8914.JPG"

# detect_text(path)
