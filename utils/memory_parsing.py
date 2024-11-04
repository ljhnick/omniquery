import re

def count_words(text):
    words = re.split(r'[ \n,.!?():"/;]+', text)
    word_count = len([word for word in words if word])
    
    return word_count


def parse_memory_to_string(memory: dict) -> str:
    filename = memory['filename']

    capture_method = memory['metadata']['capture_method']

    temporal = memory['metadata']['temporal_info']
    date_string = temporal['date_string']
    day_of_week = temporal['day_of_week']
    time_of_the_day = temporal['time_of_the_day']
    temporal_info = f'{date_string}, {day_of_week}, {time_of_the_day}'

    location = memory['metadata']['location']
    address = location.get('address', 'Unknown')

    content = memory['content']

    caption = content.get('caption', '')
    objects = content.get('objects', [])
    people = content.get('people', [])
    activities = content.get('activities', [])
    text = content.get('text', '')
    speech = content.get('speech', '')

    word_count = count_words(text)
    text_in_prompt = text if word_count < 100 else ""

    memory_string = f'''
memory_id: {filename}
capture method: {capture_method}
temporal info: {temporal_info}
location: {address}

Content: 
caption: {caption}
visible objects: {objects}
visible people: {people}
visible text: {text_in_prompt}
heard speech: {speech}
inferred activities: {activities}\n\n'''
    return memory_string

def parse_memory_to_string_lite(memory: dict) -> str:
    filename = memory['filename']

    capture_method = memory['metadata']['capture_method']

    temporal = memory['metadata']['temporal_info']
    date_string = temporal['date_string']
    day_of_week = temporal['day_of_week']
    time_of_the_day = temporal['time_of_the_day']
    temporal_info = f'{date_string}, {day_of_week}, {time_of_the_day}'

    location = memory['metadata']['location']
    address = location.get('address', 'Unknown')

    content = memory['content']

    caption = content.get('caption', '')
    objects = content.get('objects', [])
    people = content.get('people', [])
    activities = content.get('activities', [])
    text = content.get('text', '')
    speech = content.get('speech', '')

    word_count = count_words(text)
    text_in_prompt = text if word_count < 100 else ""

    memory_string = f'''
memory_id: {filename}
capture method: {capture_method}
temporal info: {temporal_info}
location: {address}

Content: 
caption: {caption}
visible objects: {objects}
visible people: {people}
visible text: {text_in_prompt}
heard speech: {speech}
inferred activities: {activities}\n\n'''
    return memory_string

def parse_composite_context_to_string(composite_context: dict) -> str:
    event_name = composite_context['event_name']
    start_date = composite_context['start_date']
    end_date = composite_context['end_date']
    location = composite_context['location']
    # is_multi_days = composite_context['is_multi_days']
    # importance = composite_context['importance']

    composite_context_string = f'''
    event_name: {event_name}
    start_date: {start_date}
    end_date: {end_date}
    location: {location}\n\n'''
    return composite_context_string

def parse_knowledge_to_string(knowledge: dict) -> str:
    knowledge_string = f'''
    knowledge: {knowledge['knowledge']}
    memory_ids: {knowledge['memory_ids']}\n\n'''
    return knowledge_string