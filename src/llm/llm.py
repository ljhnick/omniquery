from openai import OpenAI
from io import BytesIO
import base64
import json

from .prompt_templates import merge_templates_to_dict


class LLMWrapper():
    def __init__(self,
                 templates: dict = None,
                 ) -> None:
        self.templates = merge_templates_to_dict() if templates is None else templates

class OpenAIWrapper(LLMWrapper):
    def __init__(self,
                 templates: str = None,
                 model: str = 'gpt-4o-2024-08-06',
                 ) -> None:
        super().__init__(templates)
        self.llm = OpenAI()
        self.model = model

    def _generate_messages(self):
        pass

    def _restructure_result(self, result):
        print("Reformatting the JSON-like text into a correct JSON format.")
        system_prompt = "Reformat the JSON-like text into a correct JSON format. Output the json object."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": result}]
        response, result, cost = self._call_api(messages, json_mode=True, model='gpt-4o-mini')
        return result, cost

    def _call_api(self, messages, json_mode=False, model=""):
        if model == "":
            model = self.model

        if model == 'gpt-4o-2024-08-06':
            max_tokens = 16384
        else:
            max_tokens = 4096

        if json_mode:
            response = self.llm.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        else:
            response = self.llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
            )

        if model == 'gpt-4o':
            rate = 0.000005
        elif model == 'gpt-3.5-turbo-0125':
            rate = 0.0000005
        elif model == 'gpt-4o-mini':
            rate = 0.00000015
        else:
            rate = 0.000005
        prompt_token = response.usage.prompt_tokens
        generation_token = response.usage.completion_tokens
        cost = prompt_token*rate + generation_token*rate*3
        result = response.choices[0].message.content

        return response, result, cost

    def generate_visual_content(self, image):
        system_prompt = self.templates['prompt_visual_content']

        # create a base64 image
        buff = BytesIO()
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buff, format="JPEG")
        base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")
        user_prompt = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "low"}}]
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        _, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)

        return result, cost
    
    def generate_visual_content_video(self, images, speech):
        system_prompt = f"Given frames of a video and the transcribed speech, generate the following information from the video: caption of the video describing the content, visible objects, list of description of visible people, inferred activities of the media owner. Output a JSON object with key: 'caption', 'objects', 'people', 'activities'."

        user_prompt = []
        speech_prompt = {"type": "text", "text": f"transcribed speech: {speech}"}
        user_prompt.append(speech_prompt)
        for image in images:
            buff = BytesIO()
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(buff, format="JPEG")
            base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")

            entry = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "low"}}
            
            user_prompt.append(entry)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]

        _, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)

        return result, cost
    
    def generate_composite_context(self, memory_batch_text):
        system_prompt = self.templates['prompt_composite_context']

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": memory_batch_text}]
        _, result, cost = self._call_api(messages, json_mode=True)
        try:
            result = json.loads(result)
        except:
            # use gpt-4o-mini to reprocess the result to be json
            result, _ = self._restructure_result(result)
            result = json.loads(result)

        return result, cost


    def generate_events_from_content(self, content):
        system_prompt = self.templates['prompt_events']

        content_str = ""
        for key, value in content.items():
            if key == 'text':
                continue
            content_str += f"{key}: {value}\n"

        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)

        return result, cost
    
    def generate_facts_and_knowledge(self, memory_batch_text):
        system_prompt = self.templates['prompt_facts_knowledge']
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": memory_batch_text}]

        _, result, cost = self._call_api(messages, json_mode=True)
        try:
            result = json.loads(result)
        except:
            # use gpt-4o-mini to reprocess the result to be json
            result, _ = self._restructure_result(result)
            result = json.loads(result)

        return result, cost
    
    def generate_semantic_knowledge_from_content(self, metadata, content):
        system_prompt = self.templates['prompt_semantic_knowledge']

        metadata_str = ""
        metadata_str += f"Captured date: {metadata['temporal_info']['date_string']}, {metadata['temporal_info']['day_of_week']} {metadata['temporal_info']['time_of_the_day']}\n"
        try:
            metadata_str += f"Captured location: {metadata['location']['address']}\n"
        except:
            metadata_str += "Captured location: Unknown (screenshot or saved from online)\n"

        content_str = ""
        for key, value in content.items():
            content_str += f"{key}: {value}\n"

        user_prompt = metadata_str + content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)

        return result, cost
    
    def generate_acitivity_and_knowledge(self, node, events):
        system_prompt = self.templates['prompt_activity_and_knowledge']

        events_by_date = events['by_date']
        node_content = node.textualize_memory()

        example_1_prompt = "Memory: Captured time: {'date_string': '2024:05:13 21:26:17', 'day_of_week': 'Monday', 'time_of_the_day': 'Evening'} Captured location: Hawaii Prince Hotel, Holomoana Street, Ala Moana, Honolulu, Honolulu County, Hawaii, 96841, United States Capture method: photo Content: caption: A torn brown paper bag with a receipt attached, placed on a small table. objects: ['brown paper bag', 'receipt', 'small table'] people: [] text: 250. 413 HI 96815 Take Server: TOGO V #59 red: Out Table TG3, Jiahao L. 5/13/24 8:47 PM 1 Hawaiian Tuna Outlet 1 MixPoke Don 1 TOGO Mochiko Plate Subtotal Tax $26.00 $16.00 $17.00 $59.00 $2.78 $61.78 Total Suggested Tip: 18X: (Tip $10.62 Total $72.40) 20%: (Tip $11.80 Total $73.58) 2X: (Tip $12.98 Total $74.76) Tip percentages are based on the check price before taxes. BIG MAHALO!"
        example_1_output = '''{\n  "activity": "Eating takeout",\n  "knowledge": [\n    "I ordered takeout at Hawaii Prince Hotel"\n  ]\n}'''

        example_2_prompt = '''Memory:\nCaptured time: {'date_string': '2024:05:14 15:28:24', 'day_of_week': 'Tuesday', 'time_of_the_day': 'Afternoon'}\nCaptured location: Hawaii Convention Center, 1801, Kalākaua Avenue, McCully, Honolulu, Honolulu County, Hawaii, 96815, United States\nCapture method: photo\nContent:\ncaption: A person holding a conference badge with the name 'Nazar Ponochevnyi' from the University of Toronto. The badge includes a QR code and ribbons that say 'My First CHI' and 'PRESENTER'.\nobjects: ['conference badge', 'QR code', 'ribbons', 'lanyard']\npeople: ['unknown']\ntext: Professional Member\nFull Conference\nCHI2024\nSurfing the World\nNazar\nPonochevnyi\nUniversity of Toronto\nhe/him/his\nMy First CHI\nPRESENTER\n\n'''
        example_2_output = '''{\n  "activity": "Attending a conference",\n  "knowledge": [\n    "I attended the CHI2024 conference at the Hawaii Convention Center",\n    "Nazar Ponochevnyi from the University of Toronto was a presenter at the conference"\n  ]\n}'''

        example_3_prompt = '''Memory:\nCaptured time: {'date_string': '2024:05:12 09:32:40', 'day_of_week': 'Sunday', 'time_of_the_day': 'Morning'}\nCaptured location: Hawaii Prince Hotel, Holomoana Street, Ala Moana, Honolulu, Honolulu County, Hawaii, 96841, United States\nCapture method: photo\nContent:\ncaption: Aerial view of a marina with numerous boats docked, adjacent to a large body of water under a clear blue sky with scattered clouds.\nobjects: ['boats', 'marina', 'parking lot', 'ocean', 'sky', 'clouds']\npeople: []\ntext: \n\n'''
        example_3_output = '''{\n  "activity": "",\n  "knowledge": ["Hawaii Prince Hotel has aerial view of marina"]\n}'''
       
        # locate the events on the same date
        date = node.date.date()
        date = str(date)
        if date in events_by_date:
            events_list = events_by_date[date]
        
        content_str = ""
        content_str += f"Memory:\n{node_content}\n"
        # content_str += f"Events on the same date:\n"
        # for event in events_list['events']:
        #     content_str += json.dumps(event)
        #     content_str += "\n"
        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": example_1_prompt},
                    {"role": "user", "content": example_1_output},
                    {"role": "user", "content": example_2_prompt},
                    {"role": "user", "content": example_2_output},
                    {"role": "user", "content": example_3_prompt},
                    {"role": "user", "content": example_3_output},
                    {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)

        # print(content_str)
        # print(result)
        result = json.loads(result)
        # print(result)

        return result, cost


    
    def generate_events_from_multi_nodes(self, nodes):
        system_prompt = self.templates['prompt_events']

        content_str = ""
        for index, node in enumerate(nodes):
            if node.is_processed_event:
                continue
            node.is_processed_event = True
            if node.has_parent:
                continue

            content_str += f"Memory {index+1}:\n"
            content_str += node.textualize_memory()

        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)

        return result, cost
    
    def filter_events(self, events_list_by_month):
        system_prompt = self.templates['prompt_filter_events']

        content_str = ""
        for event in events_list_by_month:
            content_str += json.dumps(event)
            content_str += "\n"

        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)
        
        return result, cost
    
    def identify_query_type(self, query):
        system_prompt = self.templates['prompt_identify_query_type']
        user_prompt = "Query: " + query
        example_1_prompt = "Query: Beautifil sunset in Kona resort"
        example_1_output = "retrieval"
        example_2_prompt = "Query: What was the name of the hotel I stayed at during CHI"
        example_2_output = "question"
        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": example_1_prompt},
                    {"role": "user", "content": example_1_output},
                    {"role": "user", "content": example_2_prompt},
                    {"role": "user", "content": example_2_output},
                    {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=False, model='gpt-3.5-turbo-0125')

        return result, cost
    
    def identify_event_activity(self, query):
        system_prompt = self.templates['prompt_identify_event_act_query']
        user_prompt = "Query: " + query
        example_1_prompt = "Query: Beautifil sunset in Kona resort"
        example_1_output = "{'event': 'trip to kona', 'activity': 'watch sunset'}"
        example_2_prompt = "Query: What was the name of the hotel I stayed at during CHI"
        example_2_output = "{'event': 'CHI', 'activity': ''}"
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": example_1_prompt},
                    {"role": "user", "content": example_1_output},
                    {"role": "user", "content": example_2_prompt},
                    {"role": "user", "content": example_2_output},
                    {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=False, model='gpt-3.5-turbo-0125')
        return result, cost
    
    def filter_related_event(self, query, events):
        system_prompt = self.templates['prompt_identify_related_events']
        
        content_str = "Events: \n"
        for month in events:
            events_month = events[month]
            content_str += f"Month: {month}\n"
            for idx, event in enumerate(events_month):
                event_name = event['event_name']
                content_str += f"id: {idx}, Event name: {event_name}\n"
            content_str += '\n'
        
        content_str += f"Query: {query}"

        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)
        return result, cost
    
    def filter_nodes_related_to_query(self, query, nodes):
        system_prompt = "Given a query and a list of nodes, identify the nodes related to the query. Rate the relatedness from 1 to 3, with 3 being strongly related, 2 being possibly semantically related, and 1 being not related. Output the related nodes as a JSON object with the key 'nodes'. Each node should include the keys 'node_id', and 'relatedness'."

        content_str = "Nodes: \n"
        for idx, node in enumerate(nodes):
            content_str += f"id: {idx}, node caption: {node.content['caption']}\n"
        content_str += f"Query: {query}"

        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=False, model='gpt-3.5-turbo-0125')
        result = json.loads(result)
        return result, cost
    
    def filter_knowledge_related_to_query(self, query, knowledge):
        system_prompt = "Given a query and a list of knowledge, identify the knowledge related to the query. Rate the relatedness from 1 to 3, with 3 being strongly related and 1 being not related. Output the related knowledge as a JSON object with the key 'knowledge'. Each knowledge should include the keys 'knowledge_id', 'knowledge_name' and 'relatedness'."

        content_str = "Knowledge: \n"
        for knowledge_item in knowledge:
            knowledge_id = knowledge_item[0]
            knowledge_name = knowledge_item[1]['knowledge']
            content_str += f"id: {knowledge_id}, knowledge: {knowledge_name}\n"
        content_str += f"Query: {query}"

        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=False, model='gpt-3.5-turbo-0125')
        result = json.loads(result)
        return result, cost
    
    def generate_answer(self, query, nodes, events=None, knowledge=None):
        system_prompt = "Given a query, a list of memories, related events and personal knowledge, generate the answer to the query. Output the answer as a string. Be concise and precise."

        # system_prompt = "Given a query, a list of memories, related events and personal knowledge, generate the answer to the query. Assess whether the answer is generated from the memories, events, or knowledge (prioritize memory and knowledge). Return the out in a JSON object with the keys 'answer', 'source'. "

        content_str = "Memories:\n"
        for idx, node in enumerate(nodes):
            content_str += f"id: {idx}, memory caption: {node.textualize_memory()}\n"

        content_str += "Events:\n"
        for event in events:
            event_name = event['event_name']
            start_date = event['start_date']
            end_date = event['end_date']
            event_str = f"Event name: {event_name}, start date: {start_date}, end date: {end_date}\n"
            content_str += event_str
            if event['child_events'] != []:
                child_events = event['child_events']
                for child_event in child_events:
                    event_name = child_event['event_name']
                    start_date = child_event['start_date']
                    end_date = child_event['end_date']
                    event_str = f"Event name: {event_name}, start date: {start_date}, end date: {end_date}\n"
                    content_str += event_str
        content_str += "\n"

        if knowledge is not None:
            content_str += "Knowledge:\n"
            for k in knowledge:
                content_str += f"knowledge: {k}\n"

        content_str += f"Query: {query}"
        user_prompt = content_str
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=False)
        return result, cost
    
    def augment_query(self, query, today):
        system_prompt = self.templates['prompt_augment_query']

        user_prompt = f"Query: {query}, Today: {today}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)
        return result, cost

    def query_memory(self, query, memory_prompt):
        system_prompt = self.templates['prompt_query_memory']

        user_prompt = f"Query: {query}\n"
        user_prompt += memory_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)
        return response, result, cost
    
    def query_rag(self, query, memory_prompt):
        system_prompt = self.templates['prompt_query_rag']

        user_prompt = f"Query: {query}\n"
        user_prompt += memory_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)
        return response, result, cost
    
    def query_rag_multimodal(self, query, content):
        system_prompt = self.templates['prompt_query_rag']

        user_prompt = [{'type': 'text', 'text': query}]
        user_prompt = user_prompt + content

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)
        return response, result, cost

    def filter_related_composite_context(self, query, composite_context):
        system_prompt = self.templates['prompt_filter_related_composite_context']

        user_prompt = ""
        for context in composite_context:
            event_name = context['event_name']
            context_str = f"event_name: '{event_name}'\n"
            user_prompt += context_str

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        result = json.loads(result)
        return result, cost
    
    def chunking_text(self, text):
        system_prompt = "Given a long text, chunk the text into smaller segments. Output the list of chunks in a JSON object with the key 'chunks'."
        user_prompt = f"Text: {text}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=True)
        return result, cost

    def compare_similarity(self, text1, text2):
        system_prompt = self.templates['prompt_compare_similarity']
        user_prompt = f"text1: {text1}\ntext2: {text2}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response, result, cost = self._call_api(messages, json_mode=False, model='gpt-4o-mini')
        return result, cost
    
    def calculate_embeddings(self, text, model="text-embedding-3-small"):
        if text == "":
            return None
        return self.llm.embeddings.create(input = [text], model=model).data[0].embedding