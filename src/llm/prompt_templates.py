generate_visual_content = "Generate the following info from the image: caption that describe the image content, visible objects, descripiton of visible people, inferred activities of the media owner. Output a JSON object with key: 'caption', 'objects', 'people', 'activities'."


generate_composite_context = """Generate a list of EVENTS inferred from the given memory. Events could be inferred from the explicit content (e.g., text showing the event info) or implicit cues (e.g., multiple changes in location indicating travel). Focus on relatively important events such as travel, conferences, and important meetings and focus less on trivial events. 
For each event, identify the related episodic memory ids. This could be due to time (e.g., the memory occurs during the event), location (e.g., the memory takes place at the event location), or specific content (e.g., the memory mentions the event).
Additionally, rate the importance of each event on a scale from 1 to 3, where 3 denotes very major events (e.g., multi-day events or highly important events), 2 denotes moderately important events, and 1 denotes less important events.

Exemplar events types include:
An academic conference: "CHI 2024 Conference held in Honolulu";
Recreational travel: "Trip to Hawaii", "Trip to Big Island, Hawaii";
Locational change: "Location changed from Seattle to Irvine";
Outdoor activities: "Camping trip";
Personal milestones: "Birthday celebration", "Graduation ceremony";
etc.

Output the list of events in a JSON object with the key 'events'. Each event should be represented as a sub JSON object with the following keys: 'event_name' (detailed and concise), 'memory_ids' (list), 'start_date', 'end_date' (could be the same as start_date), 'location', 'is_multi_days', and 'importance'.
"""

generate_facts_and_knowledge = """Generate a list of FACTS or KNOWLEDGE (referred to knowledge in the following) that can be inferred from the given memory. Focus on relatively important high-level semantic knowledge and focus less on trivial events. Avoid specific details about individual media. 
Examples include: "CHI 2024 Conference is held in Honolulu, Hawaii, from May 12th to May 16th", or "Jiaju Ma's birthday is on October 17th". The knowledge should be detailed and self-contained.
Also identify the most representative episodic memories that contribute to the understanding of the knowledge. The list of episodic memories should be no more than 10 instances.
Output a JSON object with the key 'knowledge'. Each knowledge item should include 'knowledge', 'memory_ids' (list)
"""

augment_query = """Given a query and today's date, identify the contextual filters. Contextual filters may include: 
temporal information: e.g., "last week."
location information: e.g., "Hawaii."
visible objects: e.g., "poke bowl."
people Seen: e.g., "people at the conference."
activities performed: e.g., "ordering in a restaurant."
and more complex contexts such as events or travel: e.g., "traveling to Hawaii."
The query may not contain detailed contextual filters. In such cases, make reasonable inferences. For example, for query "What products did I buy from Sephora", the result could be obtained from a Sephora receipt. Thus inferred contextual filters for objects might be "makeup/skincare products or receipts."
Output a JSON object with the key 'augmented_query', including the sub-keys 'start_date', 'end_date', 'location', 'objects', 'people', 'activities', and 'complex_context'. Each sub-key should be a single string. Leave any sub-key empty if not applicable.
"""

filter_related_composite_context = """Given a list of retrieved composite context, filter out the unrelated context to the given query. Output the filtered list of composite context in a JSON object with the key 'composite_context', each composite context should include the key 'event_name'.
"""

query_memory = """Given a query, a list of memories and personal knowledge, generate a comprehensive answer to the query. Identify the episodic memories that can provide evidence to the question. The list of episodic memories should be comprehensive but no more than 30 instances. If the answer is not explicitly presented in the memories, make a reasonable inference. Output a JSON object with the key 'answer', 'explanation' and 'memory_ids'. The 'answer' should be a string and 'memory_ids' should be a list of memory ids.
"""

query_rag = """Given a query, a list of memories, generate the answer to the query. Identify the episodic memories that can provide evidence to the question. If the answer is not explicitly presented in the memories, make a reasonable inference. Output a JSON object with the key 'answer', 'explanation' and 'memory_ids'. The 'answer' should be a string and 'memory_ids' should be a list of memory ids.
"""
###  legacy


# generate_events = "Based on the context, list EVENTS inferred for the image owner, ranging from major occurrences like conferences or travel to minor activities like dining out or walking. Be precise and concise. Output the list of events in a JSON object with key: 'events'."

# generate_events = "Generate a list of EVENTS inferred from the given memory. Focus on relatively important events, such as travel, attending a conference, important meetings, etc. Also reason whether the event could span multiple days. Output the list of events in a JSON object with the key 'events'. For each event, include 'event_name', 'date', 'location', and 'is_multi_days' as keys in the sub JSON object."

generate_events = """Generate a list of EVENTS inferred from the given memory. Focus on relatively important events such as travel, conferences, and important meetings and focus less on trivial events. Assess whether each event could span multiple days. 
Additionally, rate the importance of each event on a scale from 1 to 3, where 3 denotes very major events (e.g., multi-day events or highly important events), 2 denotes moderately important events, and 1 denotes less important events.
Output the list of events in a JSON object with the key 'events'. Each event should be represented as a sub JSON object with the following keys: 'event_name', 'date', 'location', 'is_multi_days', and 'importance'.
"""



filter_events = """You are an agent with powerful processing and reasoning skills to summarize and merge events.
Merge events that span multiple days or have similar meanings and remove less important events. 
Return a JSON object with the key 'events'. Each event should include 'event_name', 'start_date', 'end_date', 'importance', and 'child_events'. The 'child_events' should have the same keys."""

generate_semantic_knowledge = """Based on the metadata and context, reason and generate high-level semantic knowledge, such as a person's birthday, in the format "X verb Y," like "Jerry's birthday is on March 2nd." 
Avoid specific details about individual media, such as the locations or actions. Focus on general semantic memory.
If an image shows a receipt, instead repeating what was on the receipt, output "I ate at a restaurant in XX during YY".
Output a list of semantic knowledge in a JSON object with the key 'semantic_knowledge'.
"""


# generate_semantic_knowledge = """Based on the metadata and context, reason and generate high-level semantic knowledge, such as a person's birthday, in the format "X verb Y", like "Jerry's birthday is on March 2nd." If no significant information is found, do not repeat the provided information. Output a list of semantic knowledge in a JSON object with the key 'semantic_knowledge'.
# """



generate_activity_and_knowledge = """You are a capable agent to infer activities and knowledge from provided memory. Extract the activities associated with the person, in a short and concise manner. If the activity is not notable, return an empty string. Infer high-level semantic knowledge from the context, focusing on high-level knowledge and avoiding episodic details. If there is no significant information, return empty.
Output the activity and knowledge in a JSON object with the keys 'activity' and 'knowledge', where 'activity' returns a string and 'knowledge' returns a list."""

identify_query_type = '''Identify the type of query as either a "retrieval" or a "question". "Retrieval" indicates the user intends to locate a specific memory, while "question" indicates the user is asking a question and seeking an answer. Output the query type only as a string'''

identify_event_act_query = '''Given a query, identify if it refers to a certain event or activity. If not, return empty. Output the result as a JSON object with key 'event' and 'activity'.'''

identify_related_events = '''Given a query and a list of events, identify the events related to the query. Rate the relatedness from 1 to 3, with 3 being strongly related. Output the related events as a JSON object with the key 'events'. Each event should include the keys 'month', 'event_id', 'event_name', and 'relatedness'.'''


compare_similarity = "Compare the similarity between the two texts. Rate from 1-10, 1 being completely different and 10 being identical. Output the similarity score only."

def merge_templates_to_dict() -> dict:
    template_dict = {
        'prompt_visual_content': generate_visual_content,
        'prompt_events': generate_events,
        'prompt_composite_context': generate_composite_context,
        'prompt_semantic_knowledge': generate_semantic_knowledge,
        'prompt_facts_knowledge': generate_facts_and_knowledge,
        'prompt_compare_similarity': compare_similarity,
        'prompt_filter_events': filter_events,
        'prompt_activity_and_knowledge': generate_activity_and_knowledge,
        'prompt_identify_query_type': identify_query_type,
        'prompt_identify_event_act_query': identify_event_act_query,
        'prompt_identify_related_events': identify_related_events,
        'prompt_augment_query': augment_query,
        'prompt_filter_related_composite_context': filter_related_composite_context,
        'prompt_query_memory': query_memory,
        'prompt_query_rag': query_rag
    }

    return template_dict