import numpy as np
import json
import os

from datetime import datetime
from tqdm import tqdm

from ..llm.llm import OpenAIWrapper

from utils.memory_parsing import parse_memory_to_string, parse_memory_to_string_lite


class AugmentContext():
    def __init__(
            self,
            memory_content_processed: list,
    ) -> None:
        self.memory_content_processed = memory_content_processed

        self.composite_context = []
        self.composite_context_embeddings = None

        self.knowledge = []
        self.knowledge_embeddings = None

        self.llm = OpenAIWrapper()
        self.cost = 0

        self.augment()

    def update_composite_list(self, event):
        event_name = event['event_name']
        emb = self.llm.calculate_embeddings(event_name)
        if self.composite_context_embeddings is None:
            self.composite_context_embeddings = np.array(emb).reshape(1, -1)
            self.composite_context.append(event)
            return
        
        # calculate the similarity
        emb_vertical = np.array(emb).reshape(-1, 1)

        similarities = np.matmul(self.composite_context_embeddings, emb_vertical)
        similarities = similarities.flatten()

        # find max similarity, if it is above a threshold, merge the events
        max_similarity = max(similarities)
        if max_similarity > 0.8:
            # print(max_similarity)
            index = np.argmax(similarities)

            # check if date can be merged
            prev_start_date = self.composite_context[index]['start_date']
            prev_end_date = self.composite_context[index]['end_date']
            this_start_date = event['start_date']
            this_end_date = event['end_date']
            if prev_end_date >= this_start_date and this_end_date > prev_start_date:
                start_date = min(prev_start_date, this_start_date)
                end_date = max(prev_end_date, this_end_date)

                self.composite_context[index]['start_date'] = start_date
                self.composite_context[index]['end_date'] = end_date
                combined_memory_ids = list(set(self.composite_context[index]['memory_ids'] + event['memory_ids']))
                self.composite_context[index]['memory_ids'] = combined_memory_ids
            else:
                self.composite_context.append(event)
                self.composite_context_embeddings = np.vstack([self.composite_context_embeddings, emb])
            # 
        else:
            self.composite_context.append(event)
            self.composite_context_embeddings = np.vstack([self.composite_context_embeddings, emb])

    def update_knowledge_list(self, knowledge):
        knowledge_name = knowledge['knowledge']
        emb = self.llm.calculate_embeddings(knowledge_name)
        if self.knowledge_embeddings is None:
            self.knowledge_embeddings = np.array(emb).reshape(1, -1)
            self.knowledge.append(knowledge)
            return
        
        # calculate the similarity
        emb_vertical = np.array(emb).reshape(-1, 1)

        similarities = np.matmul(self.knowledge_embeddings, emb_vertical)
        similarities = similarities.flatten()

        max_similarity = max(similarities)
        if max_similarity > 0.8:
            index = np.argmax(similarities)
            combined_memory_ids = list(set(self.knowledge[index]['memory_ids'] + knowledge['memory_ids']))
            self.knowledge[index]['memory_ids'] = combined_memory_ids
        else:
            self.knowledge.append(knowledge)
            self.knowledge_embeddings = np.vstack([self.knowledge_embeddings, emb])



    def detect_composite(self, memory_in_window):
        # '''
        # memory_id: <filename>
        # temporal info: <date>
        # location info: <location>
        # content: <content>
        # inferred activities: <activities>
        # \n
        # '''

        batch_memory = ''
        for memory in memory_in_window:
            memory_string = parse_memory_to_string(memory)
            batch_memory += memory_string

        try:
            result, cost = self.llm.generate_composite_context(batch_memory)
            self.cost += cost
            result_knowledge, cost = self.llm.generate_facts_and_knowledge(batch_memory)
        except Exception as e:
            print(e)
            return

        for event in result['events']:
            self.update_composite_list(event)

        for knowledge in result_knowledge['knowledge']:
            self.update_knowledge_list(knowledge)

        
    def augment_slide_window(self, step=3, window_size=5):
        memory = self.memory_content_processed

        start = 0
        next_start = 0
        end = 0

        memory_batch = []

        composite_vector_db_path = 'data/vector_db/composite_vector_db.npy'
        composite_list_path = 'data/vector_db/composite_list.json'
        knowledge_vector_db_path = 'data/vector_db/knowledge_vector_db.npy'
        knowledge_list_path = 'data/vector_db/knowledge_list.json'

        if os.path.exists(composite_vector_db_path):
            self.composite_context_embeddings = np.load(composite_vector_db_path)
            with open(composite_list_path, 'r') as f:
                self.composite_context = json.load(f)
            self.knowledge_embeddings = np.load(knowledge_vector_db_path)
            with open(knowledge_list_path, 'r') as f:
                self.knowledge = json.load(f)
            return
        
        # memory = memory[:50]

        while start < len(memory) and end < len(memory):
            start_date = memory[start]['metadata']['temporal_info']['date_string']
            start_date = datetime.strptime(start_date, "%Y:%m:%d %H:%M:%S")
            for i in range(start, len(memory)):
                this_date = memory[i]['metadata']['temporal_info']['date_string']
                this_date = datetime.strptime(this_date, "%Y:%m:%d %H:%M:%S")
                if (this_date - start_date).days >= step and next_start == start:
                    next_start = i
                if (this_date - start_date).days >= window_size:
                    end = i
                    break
                end = i+1

            memory_in_window = memory[start:end]
            
            memory_batch.append(memory_in_window)
            start = next_start

        for memory_in_window in tqdm(memory_batch):
            self.detect_composite(memory_in_window)

        # save
        np.save(composite_vector_db_path, self.composite_context_embeddings)
        with open(composite_list_path, 'w') as f:
            json.dump(self.composite_context, f, indent=4)

        np.save(knowledge_vector_db_path, self.knowledge_embeddings)
        with open(knowledge_list_path, 'w') as f:
            json.dump(self.knowledge, f, indent=4)
        

    def update_vector_db_and_list(self, category, new_element, memory_id):
        new_emb = self.llm.calculate_embeddings(new_element)
        if new_emb is None:
            return
        
        if category == 'objects':
            vector_db = self.objects_vector
            element_list = self.objects_list
        elif category == 'people':
            vector_db = self.people_vector
            element_list = self.people_list
        elif category == 'activities':
            vector_db = self.activities_vector
            element_list = self.activities_list
        else:
            return
        
        if vector_db is None or len(vector_db) == 0:
            vector_db = np.array(new_emb).reshape(1, -1)
            element_dict = {f'{category}': new_element, 'memory_ids': [memory_id]}
            element_list.append(element_dict)
            if category == 'objects':
                self.objects_vector = vector_db
                self.objects_list = element_list
            elif category == 'people':
                self.people_vector = vector_db
                self.people_list = element_list
            elif category == 'activities':
                self.activities_vector = vector_db
                self.activities_list = element_list
            return vector_db, element_list
        
        emb_vertical = np.array(new_emb).reshape(-1, 1)
        similarities = np.matmul(vector_db, emb_vertical)
        similarities = similarities.flatten()

        max_similarity = max(similarities)
        if max_similarity > 0.8:
            index = np.argmax(similarities)
            combined_memory_ids = list(set(element_list[index]['memory_ids'] + [memory_id]))
            element_list[index]['memory_ids'] = combined_memory_ids
        else:
            element_dict = {f'{category}': new_element, 'memory_ids': [memory_id]}
            element_list.append(element_dict)
            vector_db = np.vstack([vector_db, new_emb])

        if category == 'objects':
            self.objects_vector = vector_db
            self.objects_list = element_list
        elif category == 'people':
            self.people_vector = vector_db
            self.people_list = element_list
        elif category == 'activities':
            self.activities_vector = vector_db
            self.activities_list = element_list

    def augment_atomic_context(self):
        # TODO 
        # merge similar object / people / activities and save them in a vector database

        if not os.path.exists('data/vector_db'):
            os.makedirs('data/vector_db')

        objects_vector_db_path = 'data/vector_db/objects_vector_db.npy'
        people_vector_db_path = 'data/vector_db/people_vector_db.npy'
        activities_vector_db_path = 'data/vector_db/activities_vector_db.npy'

        objects_list_path = 'data/vector_db/objects_list.json'
        people_list_path = 'data/vector_db/people_list.json'
        activities_list_path = 'data/vector_db/activities_list.json'

        if os.path.exists(objects_vector_db_path):
            self.objects_vector = np.load(objects_vector_db_path)
            self.people_vector = np.load(people_vector_db_path)
            self.activities_vector = np.load(activities_vector_db_path)

            with open(objects_list_path, 'r') as f:
                self.objects_list = json.load(f)
            with open(people_list_path, 'r') as f:
                self.people_list = json.load(f)
            with open(activities_list_path, 'r') as f:
                self.activities_list = json.load(f)
            return

        self.objects_vector = None
        self.people_vector = None
        self.activities_vector = None

        self.objects_list = []
        self.people_list = []
        self.activities_list = []

        for memory in tqdm(self.memory_content_processed[:]):
            memory_id = memory['filename']
            objects = memory['content']['objects']
            people = memory['content']['people']
            activities = memory['content']['activities']

            if isinstance(objects, list):
                for obj in objects:
                    self.update_vector_db_and_list('objects', obj, memory_id)
            elif isinstance(objects, str):
                self.update_vector_db_and_list('objects', objects, memory_id)

            if isinstance(people, list):
                for person in people:
                    if isinstance(person, dict):
                        person = person.get('description', '')
                    self.update_vector_db_and_list('people', person, memory_id)
            elif isinstance(people, str):
                self.update_vector_db_and_list('people', people, memory_id)

            if isinstance(activities, list):
                for activitiy in activities:
                    self.update_vector_db_and_list('activities', activitiy, memory_id)
            elif isinstance(activities, str):
                self.update_vector_db_and_list('activities', activities, memory_id)

        # save

        np.save(objects_vector_db_path, self.objects_vector)
        np.save(people_vector_db_path, self.people_vector)
        np.save(activities_vector_db_path, self.activities_vector)

        with open(objects_list_path, 'w') as f:
            json.dump(self.objects_list, f, indent=4)
        with open(people_list_path, 'w') as f:
            json.dump(self.people_list, f, indent=4)
        with open(activities_list_path, 'w') as f:
            json.dump(self.activities_list, f, indent=4)

    def augment_location(self):
        location_vector_db_path = 'data/vector_db/location_vector_db.npy'
        location_list_path = 'data/vector_db/location_list.json'

        if os.path.exists(location_vector_db_path):
            self.location_vector_db = np.load(location_vector_db_path)
            with open(location_list_path, 'r') as f:
                self.location_list = json.load(f)
            return
        
        self.location_vector_db = None
        self.location_list = []

        for memory in tqdm(self.memory_content_processed[:]):
            memory_id = memory['filename']
            location = memory['metadata']['location'].get('address', '')

            if not location or location == '':
                continue

            emb = self.llm.calculate_embeddings(location)
            if emb is None:
                continue

            if self.location_vector_db is None:
                self.location_vector_db = np.array(emb).reshape(1, -1)
                self.location_list = [{'location': location, 'memory_ids': [memory_id]}]

            # update location list
            emb_vertical = np.array(emb).reshape(-1, 1)
            similarities = np.matmul(self.location_vector_db, emb_vertical)
            similarities = similarities.flatten()

            max_similarity = max(similarities)
            if max_similarity > 0.8:
                index = np.argmax(similarities)
                combined_memory_ids = list(set(self.location_list[index]['memory_ids'] + [memory_id]))
                self.location_list[index]['memory_ids'] = combined_memory_ids
            else:
                self.location_list.append({'location': location, 'memory_ids': [memory_id]})
                self.location_vector_db = np.vstack([self.location_vector_db, emb])
            
        # save
        np.save(location_vector_db_path, self.location_vector_db)
        with open(location_list_path, 'w') as f:
            json.dump(self.location_list, f, indent=4)
            

    def augment_text_and_speech(self):
        if not os.path.exists('data/vector_db'):
            os.makedirs('data/vector_db')
        
        text_vector_db_path = 'data/vector_db/text_vector_db.npy'
        text_list_path = 'data/vector_db/text_list.json'

        if os.path.exists(text_vector_db_path):
            self.text_vector_db = np.load(text_vector_db_path)
            with open(text_list_path, 'r') as f:
                self.text_list = json.load(f)
            return
        
        self.text_vector_db = None
        self.text_list = []

        for memory in tqdm(self.memory_content_processed[:]):
            memory_id = memory['filename']
            memory_content = memory['content']
            text = memory_content.get('text', '')
            speech = memory_content.get('speech', '')

            if text and text != '':
                try:
                    emb = self.llm.calculate_embeddings(text)
                    if emb is not None:
                        if self.text_vector_db is None:
                            self.text_vector_db = np.array(emb).reshape(1, -1)
                            self.text_list.append({'text': text, 'memory_ids': [memory_id]})
                        else:
                            self.text_vector_db = np.vstack([self.text_vector_db, emb])
                            self.text_list.append({'text': text, 'memory_ids': [memory_id]})
                except:
                    # the text might be too long, exceed the token limit
                    # chunk the text first
                    texts = self.llm.chunking_text(text)
                    texts = texts.get('chunks', [])
                    for t in texts:
                        emb = self.llm.calculate_embeddings(t)
                        if emb is not None:
                            if self.text_vector_db is None:
                                self.text_vector_db = np.array(emb).reshape(1, -1)
                                self.text_list.append({'text': t, 'memory_ids': [memory_id]})
                            else:
                                self.text_vector_db = np.vstack([self.text_vector_db, emb])
                                self.text_list.append({'text': t, 'memory_ids': [memory_id]})

            if speech and speech != '':
                emb = self.llm.calculate_embeddings(speech)
                if emb is not None:
                    if self.text_vector_db is None:
                        self.text_vector_db = np.array(emb).reshape(1, -1)
                        self.text_list.append({'text': speech, 'memory_ids': [memory_id]})
                    else:
                        self.text_vector_db = np.vstack([self.text_vector_db, emb])
                        self.text_list.append({'text': speech, 'memory_ids': [memory_id]})

        # save
        np.save(text_vector_db_path, self.text_vector_db)
        with open(text_list_path, 'w') as f:
            json.dump(self.text_list, f, indent=4)


    def generate_caption_vector_db(self):

        save_path_vector_db = 'data/vector_db/caption_vector_db.npy'
        save_path_list = 'data/vector_db/caption_list.json'

        if os.path.exists(save_path_vector_db):
            self.caption_vector_db = np.load(save_path_vector_db)
            with open(save_path_list, 'r') as f:
                self.caption_list = json.load(f)
            return

        self.caption_vector_db = None
        for memory in tqdm(self.memory_content_processed[:]):
            memory_id = memory['filename']
            caption = memory['content']['caption']

            emb = self.llm.calculate_embeddings(caption)
            if emb is None:
                continue

            if self.caption_vector_db is None:
                self.caption_vector_db = np.array(emb).reshape(1, -1)
                self.caption_list = [{'caption': caption, 'memory_ids': [memory_id]}]
                continue
            self.caption_vector_db = np.vstack([self.caption_vector_db, emb])
            self.caption_list.append({'caption': caption, 'memory_ids': [memory_id]})

        # save
        np.save(save_path_vector_db, self.caption_vector_db)
        with open(save_path_list, 'w') as f:
            json.dump(self.caption_list, f, indent=4)

    def generate_vector_db_for_rag(self):

        save_path_vector_db = 'data/vector_db/vector_db_rag.npy'
        save_path_list = 'data/vector_db/vector_db_list.json'

        if os.path.exists(save_path_vector_db):
            self.vector_db_rag = np.load(save_path_vector_db)
            with open(save_path_list, 'r') as f:
                self.vector_db_list = json.load(f)
            return
        
        self.vector_db_rag = None
        self.vector_db_list = []

        for memory in tqdm(self.memory_content_processed[:]):
            caption = memory['content']['caption']
            metadata = memory['metadata']
            temporal = f'{metadata["temporal_info"]["date_string"]}, {metadata["temporal_info"]["day_of_week"]}, {metadata["temporal_info"]["time_of_the_day"]}'
            location = metadata['location'].get('address', '')

            # indexing
            # memory_entry = f'{caption} {temporal} {location}'
            memory_entry = parse_memory_to_string_lite(memory)
            memory_id = memory['filename']
            entry = {'memory': memory_entry, 'memory_ids': [memory_id]}

            emb = self.llm.calculate_embeddings(memory_entry)
            if emb is None:
                continue
            emb = np.array(emb).reshape(1, -1)

            if self.vector_db_rag is None:
                self.vector_db_rag = emb
                continue
            self.vector_db_rag = np.vstack([self.vector_db_rag, emb])
            self.vector_db_list.append(entry)

        # save
        np.save(save_path_vector_db, self.vector_db_rag)
        with open(save_path_list, 'w') as f:
            json.dump(self.vector_db_list, f, indent=4)

    def augment(self):
        print("Indexing atomic context...")
        self.augment_atomic_context()
        self.augment_location()
        print("Indexing text and speech...")
        self.augment_text_and_speech()
        print("Indexing captions...")
        self.generate_caption_vector_db()
        print("Inferring composite context...")
        self.augment_slide_window()

        print("Indexing whole memory for RAG...")
        self.generate_vector_db_for_rag()

        # embeddings (vector db) will be saved in /vector_db folder
        # all other processed context and knowledge will be saved in the context.json file in /processed folder