import numpy as np
import json
import time


from datetime import datetime

from ..process.memory import Memory
from .query_augment import QueryAugmentation
from ..llm.llm import OpenAIWrapper

from utils.memory_parsing import parse_memory_to_string, parse_composite_context_to_string, parse_knowledge_to_string

class QueryHandler():
    def __init__(self,
                 memory: Memory):
        self.memory = memory
        self.llm = OpenAIWrapper()
        self.cost = 0
        
        self._load_augmented_memory(memory)

    def _load_augmented_memory(self, memory: Memory):
        self.memory_to_query = memory.memory_content_processed.copy()

        augmented_context = memory.augment_context

        self.caption = augmented_context.caption_list
        self.caption_vector = augmented_context.caption_vector_db

        self.text = augmented_context.text_list
        self.text_vector = augmented_context.text_vector_db

        self.objects = augmented_context.objects_list
        self.objects_vector = augmented_context.objects_vector

        self.people = augmented_context.people_list
        self.people_vector = augmented_context.people_vector

        self.activities = augmented_context.activities_list
        self.activities_vector = augmented_context.activities_vector

        self.composite_context = augmented_context.composite_context
        self.composite_context_vector = augmented_context.composite_context_embeddings

        self.knowledge = memory.augment_context.knowledge
        self.knowledge_vector = memory.augment_context.knowledge_embeddings

        self.rag = augmented_context.vector_db_list
        self.rag_vector = augmented_context.vector_db_rag

        self.location_list = augmented_context.location_list
        self.location_vector = augmented_context.location_vector_db

    def filter_date(self, start_date, end_date):
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Filter the memories
        filtered_memory = [
            memory for memory in self.memory_to_query
            if 'metadata' in memory and 'temporal_info' in memory['metadata']
            and 'date_string' in memory['metadata']['temporal_info']
            and start_date_obj <= datetime.strptime(memory['metadata']['temporal_info']['date_string'], "%Y:%m:%d %H:%M:%S") <= end_date_obj
        ]

        memory_ids = [memory['filename'] for memory in filtered_memory]
        return memory_ids


    def filter_composite_context(self, composite_context, query, composite_topk):
        if composite_context == "":
            composite_context = query
        # if complex_context != "":
            # similarity search
        emb = self.llm.calculate_embeddings(composite_context)
        emb = np.array(emb).reshape(-1, 1)
        similarities = np.matmul(self.composite_context_vector, emb).flatten()

        top_k_context = np.argsort(similarities)[-composite_topk:][::-1]
        retrieved_composite_context = [self.composite_context[i] for i in top_k_context]

        # rerank only the related memory using LLM
        # for conposite_context in retrieved_composite_context:
        result, cost = self.llm.filter_related_composite_context(query, retrieved_composite_context)
        self.cost += cost

        
        all_related_context = []
        for related_context in result['composite_context']:
            name = related_context['event_name']
            related = [context for context in retrieved_composite_context if context['event_name'] == name]

            all_related_context = all_related_context + related


        filtered_memory_list = []
        for related_context in all_related_context:
            memory_id = related_context['memory_ids']
            filtered_memory_list = list(set(filtered_memory_list + memory_id))

            # also add the time info
            start_date = related_context.get('start_date', "")
            end_date = related_context.get('end_date', "")

            if start_date != "" and end_date != "":
                filtered_memory_ids = self.filter_date(start_date, end_date)
                filtered_memory_list = list(set(filtered_memory_list + filtered_memory_ids))

        return filtered_memory_list, all_related_context

    def filter_atomic_context(self, objects, people, activities, query, atomic_topk):
        filtered_memory_list = []
        # filter objects, people, and activities
        if objects != "":
            emb = self.llm.calculate_embeddings(objects)
            emb = np.array(emb).reshape(-1, 1)
            similarities = np.matmul(self.objects_vector, emb).flatten()

            top_k_objects = np.argsort(similarities)[-atomic_topk:][::-1]
            retrieved_objects = [self.objects[i] for i in top_k_objects]

            for object in retrieved_objects:
                memory_id = object['memory_ids']
                filtered_memory_list = list(set(filtered_memory_list + memory_id))

        if people != "":
            emb = self.llm.calculate_embeddings(people)
            emb = np.array(emb).reshape(-1, 1)
            similarities = np.matmul(self.people_vector, emb).flatten()

            top_k_people = np.argsort(similarities)[-atomic_topk:][::-1]
            retrieved_people = [self.people[i] for i in top_k_people]

            for person in retrieved_people:
                memory_id = person['memory_ids']
                filtered_memory_list = list(set(filtered_memory_list + memory_id))

        if activities != "":
            emb = self.llm.calculate_embeddings(activities)
            emb = np.array(emb).reshape(-1, 1)
            similarities = np.matmul(self.activities_vector, emb).flatten()

            top_k_activities = np.argsort(similarities)[-atomic_topk:][::-1]
            retrieved_activities = [self.activities[i] for i in top_k_activities]

            for activity in retrieved_activities:
                memory_id = activity['memory_ids']
                filtered_memory_list = list(set(filtered_memory_list + memory_id))

        return filtered_memory_list
    
    def filter_knowledge(self, query, knowledge_topk):
        # filter knowledge
        emb = self.llm.calculate_embeddings(query)
        emb = np.array(emb).reshape(-1, 1)
        similarities = np.matmul(self.knowledge_vector, emb).flatten()

        top_k_knowledge = np.argsort(similarities)[-knowledge_topk:][::-1]
        retrieved_knowledge = [self.knowledge[i] for i in top_k_knowledge]

        return retrieved_knowledge
    
    def generate_prompt(self, memory_list, composite_context, filtered_knowledge):
        memory_prompt = ""
        memory_prompt += "Memories:\n"
        for memory in memory_list:
            memory_prompt += parse_memory_to_string(memory)

        memory_prompt += "Composite Context:\n"
        for context in composite_context:
            memory_prompt += parse_composite_context_to_string(context)

        memory_prompt += "Knowledge:\n"
        for knowledge in filtered_knowledge:
            memory_prompt += parse_knowledge_to_string(knowledge)

        return memory_prompt


    def query_memory(self, query: str, topk: int = 30, atomic_topk: int = 5, location_topk: int = 5, composite_topk: int = 10, knowledge_topk: int = 10, text_topk: int = 10):
        # parse the query first
        # extract: context and target question/command

        start_time_query = time.time()
        start_time = time.time()
        augment_query = QueryAugmentation(query)
        augmented_query, cost = augment_query.augment()
        self.cost += cost
        time_cost = time.time() - start_time
        print("Query augmentation time cost: ", time_cost)
        
        # check if temporal info exists
        # if not, check if composite context exists, and if that is strict enough

        augmented_query = augmented_query['augmented_query']

        print(augmented_query)

        start_date = augmented_query['start_date']
        end_date = augmented_query['end_date']
        location = augmented_query['location']
        objects = augmented_query['objects']
        people = augmented_query['people']
        activities = augmented_query['activities']
        complex_context = augmented_query['complex_context']

        start_time = time.time()

        ####################### filter date strict
        strict_filtered_memory = []
        if start_date != "" and end_date != "":
            # Filter the memories
            strict_filtered_memory = self.filter_date(start_date, end_date)


        filtered_memory_list = []
        ####################### filter composite context
        composite_memory_list, all_related_composite = self.filter_composite_context(complex_context, query, composite_topk)
        filtered_memory_list = list(set(filtered_memory_list + composite_memory_list))

        ####################### atomic
        atomic_memory_list = self.filter_atomic_context(objects, people, activities, query, atomic_topk)
        filtered_memory_list = list(set(filtered_memory_list + atomic_memory_list))

        ####################### filter location
        if location != "":
            emb = self.llm.calculate_embeddings(location)
            emb = np.array(emb).reshape(-1, 1)
            similarities = np.matmul(self.location_vector, emb).flatten()

            top_k_location = np.argsort(similarities)[-location_topk:][::-1]
            retrieved_location = [self.location_list[i] for i in top_k_location]

            for location in retrieved_location:
                memory_id = location['memory_ids']
                filtered_memory_list = list(set(filtered_memory_list + memory_id))


        ######################@ filter caption
        query_emb = self.llm.calculate_embeddings(query)
        query_emb = np.array(query_emb).reshape(-1, 1)
        similarities = np.matmul(self.caption_vector, query_emb).flatten()

        top_k_caption = np.argsort(similarities)[-topk:][::-1]
        retrieved_caption = [self.caption[i] for i in top_k_caption]

        for caption in retrieved_caption:
            memory_id = caption['memory_ids']
            filtered_memory_list = list(set(filtered_memory_list + memory_id))


        ####################### filter text
        similarities = np.matmul(self.text_vector, query_emb).flatten()
        top_k_text = np.argsort(similarities)[-text_topk:][::-1]
        retrieved_text = [self.text[i] for i in top_k_text]

        for text in retrieved_text:
            memory_id = text['memory_ids']
            filtered_memory_list = list(set(filtered_memory_list + memory_id))



        ######################
        if strict_filtered_memory:
            # only keep the memories that are in both lists
            filtered_memory_list = list(set(filtered_memory_list) & set(strict_filtered_memory))


        ####################### identify related knowledge
        filtered_knowledge = self.filter_knowledge(query, knowledge_topk)

        # send the filtered memory to the LLM for answer
        memories_final = []
        for memory_id in filtered_memory_list:
            memory = self._search_memory_id(memory_id)
            if not memory:
                continue
            memories_final.append(memory)

        # order the memories by the date
        memories_final = sorted(memories_final, key=lambda x: x['metadata']['temporal_info']['date_string'])


        # generate prompt
        final_prompt = self.generate_prompt(memories_final, all_related_composite, filtered_knowledge)

        time_cost = time.time() - start_time
        print("Memory filtering time cost: ", time_cost)

        start_time = time.time()
        response, result, cost = self.llm.query_memory(query, final_prompt)

        tokens = response.usage.total_tokens
        print("Total tokens: ", tokens)
        self.cost += cost
        time_cost = time.time() - start_time
        print("Answer generation time cost: ", time_cost)

        time_cost_query = time.time() - start_time_query
        print("Total time cost: ", time_cost_query)

        print("API cost: ", cost)

        # print(result)
        return result
        
    def query_rag(self, query, topk=25):
        query_emb = self.llm.calculate_embeddings(query)
        query_emb = np.array(query_emb).reshape(-1, 1)
        similarities = np.matmul(self.rag_vector, query_emb).flatten()

        top_k = np.argsort(similarities)[-topk:][::-1] if len(similarities) > topk else np.argsort(similarities)[::-1]
        # retrieved_rag = [self.rag[i] for i in top_k]
        retrieved_rag = []
        for index in top_k:
            if index < len(self.rag):
                retrieved_rag.append(self.rag[index])

        prompt = ""
        for rag in retrieved_rag:
            # prompt += f'memory_id: {rag['memory_ids']}'
            memory_id = rag['memory_ids']
            prompt += f'memory_id: {memory_id} '
            prompt += f'memory: {rag["memory"]}\n'

        response, result, cost = self.llm.query_rag(query, prompt)

        print("RAG API cost: ", cost)

        # messages = []

        # from pipeline import load_memory_media

        # for rag in retrieved_rag:
        #     memory_id = rag['memory_ids'][0]

        #     text_prompt = f'memory_id: {memory_id} memory: {rag["memory"]}\n'
        #     text_prompt_msg = {'type': 'text', 'text': text_prompt}

        #     memory = self._search_memory_id(memory_id)
        #     memory_path = memory['filepath']
        #     # memory_type = memory['media_type']

        #     # load memory as base64      
        #     base64_img = load_memory_media(memory_path, self.memory)
        #     base64_img = base64_img['image']
        #     image_prompt_msg = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "low"}}
        #     messages.append(text_prompt_msg)
        #     messages.append(image_prompt_msg)

        # response, result, cost = self.llm.query_rag_multimodal(query, messages)

        return result




    def _search_memory_id(self, memory_id):
        for memory in self.memory_to_query:
            if memory['filename'] == memory_id:
                return memory


    # def similarity_search(self, augmented_query):
    #     # for each memory, calculate the similarity
    #     # return the top k memories

    #     caption = augmented_query['contextual_filter']['caption']
    #     emb = self.llm.calculate_embeddings(caption)

    #     emb = np.array(emb).reshape(-1, 1)
    #     similarities = np.matmul(self.caption_vector, emb).flatten()

    #     top_k = np.argsort(similarities)[-10:][::-1]
    #     caption_memories = [self.caption[i] for i in top_k]

    #     print(10)

        
        