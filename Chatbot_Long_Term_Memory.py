import sys
import os
import json
import time
import datetime as dt
from datetime import datetime
from uuid import uuid4
import requests
import shutil
import importlib
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np
import re
import keyboard
import traceback
import asyncio
import aiofiles
import aiohttp
import base64
import asyncio
import concurrent.futures
from colorama import Fore, Style, init
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



with open('./Settings.json', 'r') as file:
    settings = json.load(file)
Debug_Output = settings.get('Debug_Output', 'False')
Memory_Output = settings.get('Memory_Output', 'False')


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
       return file.read().strip()

def timestamp_func():
    try:
        return time.time()
    except:
        return time()

def is_url(string):
    return string.startswith('http://') or string.startswith('https://')

def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str
      
def import_api_function():
    settings_path = './Settings.json'
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    api_module_name = settings['API']
    module_path = f'./Resources/API_Calls/{api_module_name}.py'
    spec = importlib.util.spec_from_file_location(api_module_name, module_path)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    llm_api_call = getattr(api_module, 'LLM_API_Call', None)
    agent_llm_api_call = getattr(api_module, 'Agent_LLM_API_Call', None)
    input_expansion_api_call = getattr(api_module, 'Input_Expansion_API_Call', None)
    domain_selection_api_call = getattr(api_module, 'Domain_Selection_API_Call', None)
    db_prune_api_call = getattr(api_module, 'DB_Prune_API_Call', None)
    inner_monologue_api_call = getattr(api_module, 'Inner_Monologue_API_Call', None)
    intuition_api_call = getattr(api_module, 'Intuition_API_Call', None)
    final_response_api_call = getattr(api_module, 'Final_Response_API_Call', None)
    short_term_memory_response_api_call = getattr(api_module, 'Short_Term_Memory_API_Call', None)
    if llm_api_call is None:
        raise ImportError(f"LLM_API_Call function not found in {api_module_name}.py")
    return llm_api_call, agent_llm_api_call, input_expansion_api_call, domain_selection_api_call, db_prune_api_call, inner_monologue_api_call, intuition_api_call, final_response_api_call, short_term_memory_response_api_call

def load_format_settings(backend_model):
    file_path = f'./Model_Formats/{backend_model}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            formats = json.load(file)
    else:
        formats = {
            "heuristic_input_start": "",
            "heuristic_input_end": "",
            "system_input_start": "",
            "system_input_end": "",
            "user_input_start": "", 
            "user_input_end": "", 
            "assistant_input_start": "", 
            "assistant_input_end": ""
        }
    return formats

def set_format_variables(backend_model):
    format_settings = load_format_settings(backend_model)
    heuristic_input_start = format_settings.get("heuristic_input_start", "")
    heuristic_input_end = format_settings.get("heuristic_input_end", "")
    system_input_start = format_settings.get("system_input_start", "")
    system_input_end = format_settings.get("system_input_end", "")
    user_input_start = format_settings.get("user_input_start", "")
    user_input_end = format_settings.get("user_input_end", "")
    assistant_input_start = format_settings.get("assistant_input_start", "")
    assistant_input_end = format_settings.get("assistant_input_end", "")

    return heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end

def format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, response):
    try:
        if response is None:
            return "ERROR WITH API"

        if assistant_input_start == "" and assistant_input_end == "":
            return response.strip()

        botname_check = f"{botnameupper}:"
        
        if backend_model == "Llama_3":
            assistant_input_start = "assistant"
            assistant_input_end = "assistant"

        if assistant_input_start:
            while response and (response.startswith(assistant_input_start) or 
                                response.startswith('\n') or 
                                response.startswith(' ') or 
                                response.startswith(botname_check)):
                if response.startswith(assistant_input_start):
                    response = response[len(assistant_input_start):]
                elif response.startswith(botname_check):
                    response = response[len(botname_check):]
                elif response.startswith('\n') or response.startswith(' '):
                    response = response[1:]
                response = response.strip()

        botname_check = f"{botnameupper}: "
        if response.startswith(botname_check):
            response = response[len(botname_check):].strip()

        if backend_model == "Llama_3":
            if "assistant\n" in response:
                index = response.find("assistant\n")
                response = response[:index].strip()

        if assistant_input_end and response.endswith(assistant_input_end):
            response = response[:-len(assistant_input_end)].strip()

        return response
    except Exception as e:
        print(f"Error encountered: {e}")
        return response

    
async def load_filenames_and_descriptions(folder_path, username, user_id, bot_name):
    """
    Load all Python filenames in the given folder along with their descriptions.
    Returns a dictionary mapping filenames to their descriptions.
    """
    filename_description_map = {}
    
    def extract_function_code(code, func_name):
        """
        Extract the function definition from the code.
        """
        lines = code.splitlines()
        func_lines = []
        inside_func = False
        indent_level = None
        
        for line in lines:
            if inside_func:
                if line.startswith(" " * indent_level) or line.strip() == "":
                    func_lines.append(line)
                else:
                    break
            elif line.strip().startswith(f"def {func_name}("):
                inside_func = True
                indent_level = len(line) - len(line.lstrip())
                func_lines.append(line)
        
        return "\n".join(func_lines) if func_lines else None

    try:
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.py')]
        
        for filename in filenames:
            base_filename = os.path.splitext(filename)[0]
            module_path = os.path.join(folder_path, filename)
            
            with open(module_path, 'r') as file:
                code = file.read()
            
            description_function_name = f"{base_filename}_Description"
            description_function_code = extract_function_code(code, description_function_name)
            
            description = "Description function not found."
            if description_function_code:
                try:
                    local_scope = {}
                    exec(description_function_code, globals(), local_scope)
                    description = local_scope[description_function_name](username, bot_name)
                    description = description.replace('<<USERNAME>>', username)
                    description = description.replace('<<BOTNAME>>', bot_name)
                except Exception as e:
                    print(f"An error occurred: {e}")
            filename_description_map[filename] = {"filename": filename, "description": description}
                
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return filename_description_map
    
    
    
async def load_folder_names_and_descriptions(folder_path):
    """
    Load all folder names in the given folder along with their descriptions from a .txt file.
    Returns a dictionary mapping folder names to their descriptions.
    """
    folder_description_map = {}
    try:
        folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for folder in folders:
            description_file_path = os.path.join(folder_path, folder, f"{folder}.txt")
            if os.path.isfile(description_file_path):
                with open(description_file_path, 'r') as file:
                    description = file.read().strip()
            else:
                description = "Description file not found."
            folder_description_map[folder] = description
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return folder_description_map

    
def write_dataset_simple(backend_model, user_input, output):
    data = {
        "input": user_input,
        "output": output
    }
    try:
        with open(f'{backend_model}_simple_dataset.json', 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(f'{backend_model}_simple_dataset.json', 'w') as file:
            json.dump([data], file, indent=4)
            

def find_base64_encoded_json(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    pattern = re.compile(b'(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?')
    
    matches = pattern.findall(binary_data)
    valid_json_objects = []
    
    for match in matches:
        if len(match) % 4 != 0:
            continue
        try:
            decoded_data = base64.b64decode(match, validate=True)
            decoded_str = decoded_data.decode('utf-8')
            json_data = json.loads(decoded_str)
            if isinstance(json_data, dict) and 'spec' in json_data:
                valid_json_objects.append(json_data)
        except (base64.binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return valid_json_objects
    
    
def colorize_text(text):
    paragraphs = text.split('\n')
    colored_paragraphs = []
    inside_quotes = False  
    
    for paragraph in paragraphs:
        result = ""
        i = 0
        while i < len(paragraph):
            char = paragraph[i]
            if char in ['"', '“', '”']:
                if inside_quotes:
                    result += f"{char}{Style.RESET_ALL}" 
                    inside_quotes = False
                else:
                    result += f"{Fore.LIGHTYELLOW_EX}{char}" 
                    inside_quotes = True
            elif char == '*':
                result += f"{Style.RESET_ALL}" if inside_quotes else ""
            else:
                if inside_quotes:
                    result += f"{Fore.LIGHTYELLOW_EX}{char}"  
                else:
                    result += f"{Fore.LIGHTWHITE_EX}{char}" 
            i += 1
        result += Style.RESET_ALL  
        colored_paragraphs.append(result)
        
    return '\n'.join(colored_paragraphs)


class MainConversation:
    def __init__(self, username, user_id, bot_name, max_entries):
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        vector_db = settings.get('Vector_DB', 'Qdrant_DB')
        db_upload_module_name = f'Resources.DB_Upload.{vector_db}'
        self.db_upload_module = importlib.import_module(db_upload_module_name)
        self.client = self.db_upload_module.initialize_client()
        self.bot_name = bot_name
        self.username = username
        self.user_id = user_id
        self.API = settings.get('API', 'Oobabooga')
        self.backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        self.format_config = self.initialize_format()
        Use_Char_Card = settings.get('Use_Character_Card', 'False')
        Char_File_Name = settings.get('Character_Card_File_Name', 'Aetherius')
        self.LLM_API_Call, self.Agent_LLM_API_Call, self.Input_Expansion_API_Call, self.Domain_Selection_API_Call, self.DB_Prune_API_Call, self.Inner_Monologue_API_Call, self.Intuition_API_Call, self.Final_Response_API_Call, self.Short_Term_Memory_API_Call = import_api_function()
        self.max_entries = int(max_entries)
        if Use_Char_Card == "True":
            json_objects = find_base64_encoded_json(f'Characters/{Char_File_Name}.png')
            if json_objects:
                json_data = json_objects[0]
                self.bot_name = json_data['data']['name']
            else:
                print("No valid embedded JSON data found in the image.")
                Char_File_Name = bot_name
        self.botnameupper = bot_name.upper()
        self.usernameupper = username.upper()
        self.file_path = f'./History/{self.user_id}/{self.bot_name}_Conversation_History.json'
        self.summary_entries = []  
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    self.running_conversation = data.get('running_conversation', [])
                    self.summary_entries = data.get('summary_entries', [])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {self.file_path}. The file may be corrupted or improperly formatted.")
                    self.running_conversation = []
        else:
            self.running_conversation = []
            self.save_to_file()


    def initialize_format(self):
        file_path = f'./Model_Formats/{self.backend_model}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                formats = json.load(file)
        else:
            formats = {
                "user_input_start": "", 
                "user_input_end": "", 
                "assistant_input_start": "", 
                "assistant_input_end": ""
            }
        return formats

    def format_entry(self, user_input, response):
        user = f"{user_input}"
        bot = f"{response}"
        return {'user': user, 'bot': bot}

    async def append(self, timestring, user_input, response):
        entry = self.format_entry(f"[{timestring}] - {user_input}", response)
        self.running_conversation.append(entry)

        if len(self.running_conversation) > self.max_entries:
            await self.handle_overflow()

        self.save_to_file()



    async def handle_overflow(self):
        try:
            oldest_seven_entries = self.running_conversation[:7]
            summary = await self.conversation_summarizer(oldest_seven_entries)

            entries_to_remove = self.running_conversation[:5]
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                loop.run_in_executor(executor, lambda: asyncio.run(self.transitional_memory_generation(entries_to_remove)))

            self.running_conversation = self.running_conversation[5:]
            self.summary_entries.append(summary)
            self.save_to_file()

        except Exception as e:
            print("\n_________ERROR DURING OVERFLOW HANDLING________\n")
            traceback.print_exc()



    async def conversation_summarizer(self, entries):
        try:
            # <task> Improve Summary Module
            summarization = []
            summarization.append({'role': 'system', 'content': "You are an AI summarization module. Summarize the first 5 input-response pairs into a single paragraph. Ignore the last two entries; they are only for context."})
            summarization.append({'role': 'assistant', 'content': "Please provide the entries. I will summarize the first 5 into a paragraph."})
            summarization.append({'role': 'user', 'content': f"<Start Input Response Pairs>\nINPUT-RESPONSE PAIRS: {entries}</End Input Response Pairs>"})
            summarization.append({'role': 'assistant', 'content': "Here is a summary of the first five input-response pairs: <Start Summary>"})
            if self.API in ["OpenAi", "Oobabooga", "KoboldCpp"]:
                generated_summary = await self.Short_Term_Memory_API_Call(self.API, self.backend_model, summarization, self.user_id, self.bot_name)
            elif self.API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in summarization])
                generated_summary = await self.Short_Term_Memory_API_Call(self.API, prompt, self.user_id, self.bot_name)
            print(f"GENERATED SUMMARY: {generated_summary}")
            return {'role': 'assistant', 'content': generated_summary}

        except Exception as e:
            print("\n_________ERROR DURING SUMMARIZATION________\n")
            traceback.print_exc()
            return {'role': 'assistant', 'content': "Summarization failed."}

    async def transitional_memory_generation(self, entries):
        print("\n\nPlease wait. Generating Memories...\n\n")
        try:
            memory_type = "Episodic"
            episodic_list = []
            episodic_list.append({'role': 'system', 'content': f"MAIN SYSTEM PROMPT: You are a sub-module of {self.bot_name}, an AI entity designed for autonomous interaction. Your specialized function is to distill each conversation with {self.username} into a single, short and concise narrative paragraph. This paragraph should serve as {self.bot_name}'s autobiographical memory of the conversation, capturing the most significant events, context, and emotions experienced by either {self.bot_name} or {self.username}. Note that 'autobiographical memory' refers to a detailed recollection of a specific event, often including emotions and sensory experiences. Your task is to focus on preserving the most crucial elements without omitting key context or feelings. Do not include the original messages or the response pairs in your output."})

            episodic_list.append({'role': 'assistant', 'content': f"Please now generate an autobiographical memory for {self.bot_name} based on the distilled interaction."})

            episodic_list.append({'role': 'assistant', 'content': f"{self.botnameupper}'S THIRD-PERSON AUTOBIOGRAPHICAL MEMORY: <Start Episodic Memory>"})

            
            if self.API in ["OpenAi", "Oobabooga", "KoboldCpp"]:
                generated_episodic_memory = await self.Short_Term_Memory_API_Call(self.API, self.backend_model, episodic_list, self.user_id, self.bot_name)
            elif self.API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in episodic_list])
                generated_episodic_memory = await self.Short_Term_Memory_API_Call(self.API, prompt, self.user_id, self.bot_name)
            if Memory_Output == "True":
                print(f"\nGENERATED EPISODIC MEMORY: {generated_episodic_memory}")
            
            collection_name = f"BOT_NAME_{self.bot_name}"
            upload_episodic = self.db_upload_module.upload_memory(collection_name, memory_type, self.bot_name, self.user_id, generated_episodic_memory, domain=None)
        except Exception as e:
            print("\n_________EPISODIC MEMORY UPLOAD FAIL________\n")
            traceback.print_exc()
            
        try:
            memory_type = "Semantic"
            semantic_list = []

            semantic_list.append({'role': 'system', 'content': "You are a specialized AI sub-module designed to extract and record factual information from user conversations. Your task is to create 'semantic memory' entries, which represent context-independent facts, concepts, and general knowledge. These memories are only for recording knowledge, and should not record any context from the conversation.  Each bullet point you produce must be a fully self-contained, standalone statement that is clear and complete on its own. The entry should include all necessary information within itself, with no references or dependencies on other bullet points or the conversation. Each entry should follow this format: '• [Topic]: [Information]'. If no factual information is found, simply output: 'No Memories'. Do not generate any other text."})
            semantic_list.append({'role': 'assistant', 'content': "Please input the conversation text and the final user response for processing."})
            semantic_list.append({'role': 'user', 'content': f"CONVERSATION INPUT: {entries}\n\nExtract and list the factual knowledge mentioned, ensuring each bullet point is a fully independent, standalone statement with a clear topic. Make sure each bullet point includes all the necessary context to be understood on its own. If no factual information is available, respond only with: 'No Memories'."})
            semantic_list.append({'role': 'assistant', 'content': "EXTRACTED SEMANTIC MEMORY: "})

            if self.API in ["OpenAi", "Oobabooga", "KoboldCpp"]:
                generated_semantic_memory = await self.Short_Term_Memory_API_Call(self.API, self.backend_model, semantic_list, self.user_id, self.bot_name)
            elif self.API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in semantic_list])
                generated_semantic_memory = await self.Short_Term_Memory_API_Call(self.API, prompt, self.user_id, self.bot_name)
            if "No Memories" not in generated_semantic_memory:
                segments = re.split(r'(?:•|-|\n.*[•-])', generated_semantic_memory)
                if Memory_Output == "True":
                    print("\n\nGenerated Semantic Memories:")
                for segment in segments:
                    if segment.strip() == '':  
                        continue  
                    else:
                        domain_extraction = []
                        domain_extraction.append({'role': 'system', 'content': f"""You are a Domain Ontology Specialist tasked with extracting a single, generalized knowledge domain from a given text or user query. Your role is crucial for accurately categorizing and sorting database queries.

                        Instructions:
                        1. Analyze the provided text carefully.
                        2. Identify the most relevant, overarching knowledge domain that best encapsulates the main topic or theme of the text.
                        3. Express this domain using only one word.
                        4. Choose extremely general categories such as "history," "science," "technology," "arts," "sports," "politics," "health," "business," "education," "entertainment," "relationships," "travel," "food," "nature," or similar broad domains.
                        5. Ensure your response consists of only this single word, without any additional explanation, punctuation, or commentary.

                        Examples:
                        - For a query about World War II, respond with: history
                        - For a question about photosynthesis, respond with: science
                        - For a text about smartphone features, respond with: technology

                        Remember: Your entire response must be a single word representing the most general, relevant knowledge domain."""})

                        # <task> Improve module by feeding previously added domains to avoid bloat
                        domain_extraction.append({'role': 'user', 'content': f"Analyze the following text and provide the single most relevant, generalized knowledge domain:\n{segment}"})
                        if self.API in ["OpenAi", "Oobabooga", "KoboldCpp"]:
                            domain = await self.Short_Term_Memory_API_Call(self.API, self.backend_model, domain_extraction, self.user_id, self.bot_name)
                        elif self.API == "AetherNode":
                            prompt = ''.join([message_dict['content'] for message_dict in domain_extraction])
                            domain = await self.Short_Term_Memory_API_Call(self.API, prompt, self.user_id, self.bot_name)
                        domain = domain.strip()
                        if Memory_Output == "True":
                            print(f"[-DOMAIN: {domain}\nMEMORY: {segment}]")
                        
                        collection_name = f"BOT_NAME_{self.bot_name}"
                        upload_semantic = self.db_upload_module.upload_memory(collection_name, memory_type, self.bot_name, self.user_id, segment, domain)
        except Exception as e:
            print("\n_________EPISODIC MEMORY UPLOAD FAIL________\n")
            traceback.print_exc()
            


    def save_to_file(self):
        data_to_save = {
            'summary_entries': self.summary_entries,
            'running_conversation': self.running_conversation
        }
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def get_conversation_history(self):
        formatted_history = []
        if self.summary_entries:
            formatted_history.append(self.summary_entries[-1]['content'])

        for entry in self.running_conversation: 
            user_entry = entry['user']
            bot_entry = entry['bot']
            formatted_history.append(user_entry)
            formatted_history.append(bot_entry)
        return '\n'.join(formatted_history)
        
    def get_dict_conversation_history(self):
        formatted_history = []
        if self.summary_entries:
            formatted_history.append(self.summary_entries[-1])

        for entry in self.running_conversation:
            formatted_history.append({'role': 'user', 'content': entry['user']})
            formatted_history.append({'role': 'assistant', 'content': entry['bot']})
        return formatted_history

    def get_dict_formatted_conversation_history(self, user_input_start, user_input_end, assistant_input_start, assistant_input_end):
        formatted_history = []
        if self.summary_entries:
            formatted_history.append(self.summary_entries[-1])

        for entry in self.running_conversation:
            formatted_history.append({'role': 'user', 'content': f"{user_input_start}{entry['user']}{user_input_end}"})
            formatted_history.append({'role': 'assistant', 'content': f"{assistant_input_start}{entry['bot']}{assistant_input_end}"})
        return formatted_history

    def get_last_entry(self):
        if self.running_conversation:
            return self.running_conversation[-1]
        return None
    
    def delete_conversation_history(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            self.running_conversation = []
            self.summary_entries = [] 
            self.save_to_file()




async def NPC_Chatbot(user_input, username, user_id, bot_name, main_conversation, image_path=None):
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    API = settings.get('API', 'Oobabooga')
    conv_length = settings.get('Conversation_Length', '3')
    backend_model = settings.get('Model_Backend', 'Llama_3')
    LLM_Model = settings.get('LLM_Model', 'Oobabooga')
    Web_Search = settings.get('Search_Web', 'False')
    Write_Dataset = settings.get('Write_To_Dataset', 'False')
    Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Simple')
    Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
    vector_db = settings.get('Vector_DB', 'Qdrant_DB')
    LLM_API_Call, Agent_LLM_API_Call, Input_Expansion_API_Call, Domain_Selection_API_Call, DB_Prune_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
    heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
    end_prompt = ""
    Use_Char_Card = settings.get('Use_Character_Card', 'False')
    Char_File_Name = settings.get('Character_Card_File_Name', 'Aetherius')
    if Use_Char_Card != "True":
        base_path = "./Chatbot_Prompts"
        base_prompts_path = os.path.join(base_path, "Base")
        user_bot_path = os.path.join(base_path, user_id, bot_name)  
        if not os.path.exists(user_bot_path):
            os.makedirs(user_bot_path)
        prompts_json_path = os.path.join(user_bot_path, "prompts.json")
        base_prompts_json_path = os.path.join(base_prompts_path, "prompts.json")
        if not os.path.exists(prompts_json_path) and os.path.exists(base_prompts_json_path):
            async with aiofiles.open(base_prompts_json_path, 'r') as base_file:
                base_prompts_content = await base_file.read()
            async with aiofiles.open(prompts_json_path, 'w') as user_file:
                await user_file.write(base_prompts_content)
        async with aiofiles.open(prompts_json_path, 'r') as file:
            prompts = json.loads(await file.read())
        main_prompt = prompts["main_prompt"].replace('<<NAME>>', bot_name)
        greeting_msg = prompts["greeting_prompt"].replace('<<NAME>>', bot_name)
    botnameupper = bot_name.upper()
    usernameupper = username.upper()
    enable_user_profile = settings.get('Enable_User_Profile', 'True')
    enable_bot_profile = settings.get('Enable_Chatbot_Profile', 'True')
    enable_rpg_modules = settings.get('Enable_RPG_Modules', 'True')
    if enable_user_profile == "True":
        try:
            file_path = f'./Profiles/User/{user_id}.txt'
            with open(file_path, 'r') as file:
                user_profile = file.read()
        except:
            user_profile = "No User Profile Available"
    if enable_bot_profile == "True":
        try:
            file_path = f'./Profiles/User/{user_id}.txt'
            with open(file_path, 'r') as file:
                bot_profile = file.read()
        except:
            user_profile = "No Chatbot Profile Available"
    if enable_rpg_modules == "True":
        try:
            file_path = f'./RPG_Modules/Status_Effects/{bot_name}/{user_id}.txt'
            with open(file_path, 'r') as file:
                status_effects = file.read()
        except:
            status_effects = "No Current Status Effects"
    if Use_Char_Card == "True":
        json_objects = find_base64_encoded_json(f'Characters/{Char_File_Name}.png')
        if json_objects:
            json_data = json_objects[0]
            bot_name = json_data['data']['name']
            botnameupper = bot_name.upper()
            greeting_msg = json_data['data']['first_mes']
            system_prompt = json_data['data']['system_prompt']
            personality = json_data['data']['personality']
            description = json_data['data']['description']  
            scenario = json_data['data']['scenario']
            example_format = json_data['data']['mes_example']
            greeting_msg = greeting_msg.replace("{{user}}", username).replace("{{char}}", bot_name)
            system_prompt = system_prompt.replace("{{user}}", username).replace("{{char}}", bot_name)
            personality = personality.replace("{{user}}", username).replace("{{char}}", bot_name)
            description = description.replace("{{user}}", username).replace("{{char}}", bot_name)
            scenario = scenario.replace("{{user}}", username).replace("{{char}}", bot_name)
            example_format = example_format.replace("{{user}}", username).replace("{{char}}", bot_name)
            if len(example_format) > 3:
                new_prompt = f"{system_prompt}\nUse the following format:{example_format}"
            else:
                new_prompt = system_prompt
                
            main_prompt = f"{scenario}\n{personality}\n{description}"
            end_prompt = json_data['data']['post_history_instructions']
            character_tags = json_data['data']['tags']
            author_notes = json_data['data']['creator_notes']
            bot_profile = ""
        else:
            print("No valid embedded JSON data found in the image.")
            Char_File_Name = bot_name
            bot_profile = ""
    collection_name = f"BOT_NAME_{bot_name}"
    while True:
        try:
        
            conversation_history = main_conversation.get_dict_conversation_history()
            con_hist = main_conversation.get_conversation_history()
            last_con_entry = main_conversation.get_last_entry()
            timestamp = timestamp_func()
            timestring = timestamp_to_datetime(timestamp)
            
            input_expansion = []
            input_expansion.append({'role': 'system', 'content': """You are a context expander for database queries. Your sole task is to process the user's input and, if necessary, expand it using relevant context from the conversation history. Follow these rules strictly:

            1. If the user's input is clear and complete, return it exactly as is.
            2. If the input lacks context, add ONLY the minimum necessary information from the conversation history to make it a complete query.
            3. Do not add any explanations, comments, or extraneous text.
            4. The output must be a single, coherent sentence or phrase suitable for database searching.
            5. Do not use phrases like 'Based on the conversation' or 'The user is asking about' in your output.
            6. Maintain the user's original phrasing as much as possible.

            Your output will be used directly for database searching, so clarity and relevance are crucial."""})

            input_expansion.append({'role': 'assistant', 'content': f"PREVIOUS MESSAGE: {last_con_entry}\n\nPlease now provide the current User Input."})
            input_expansion.append({'role': 'user', 'content': f"CURRENT USER INPUT: {user_input}"})
            input_expansion.append({'role': 'assistant', 'content': "CONTEXT EXPANDER: Sure, here is the rephased User Input: "})

            
            if len(conversation_history) > 1:
                if API == "OpenAi":
                    expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
                if API == "Oobabooga":
                    expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
                if API == "KoboldCpp":
                    expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
                if API == "AetherNode":
                    prompt = ''.join([message_dict['content'] for message_dict in input_expansion])
                    expanded_input = await Input_Expansion_API_Call(API, prompt, username, bot_name)
                expanded_input = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, expanded_input)
                if Debug_Output == "True":
                    print(f"\n\nEXPANDED INPUT: {expanded_input}\n\n")
            else:
                expanded_input = user_input
                
            def remove_duplicate_dicts(input_list):
                seen = {}
                for index, item in enumerate(input_list):
                    seen[item] = index 
                output_list = [item for item, idx in sorted(seen.items(), key=lambda x: x[1])]
                
                return output_list
                

            # <task> Create Emotional Memory 
            db_search_module_name = f'Resources.DB_Search.{vector_db}'
            db_search_module = importlib.import_module(db_search_module_name)
            client = db_search_module.initialize_client()
            memory_type = "Episodic"
            episodic_search = db_search_module.search_db(collection_name, bot_name, user_id, expanded_input, memory_type, selected_domain=None, search_number=25)
            if Debug_Output == "True":
                print(f"\n\nEPISODIC MEMORY: {episodic_search}\n\n")
            
            semantic_search = "No Memories"
            if len(episodic_search) > 0:
                memory_type = "Semantic"
                
                domain_search = db_search_module.retrieve_domain_list(collection_name, bot_name, user_id, expanded_input, search_number=25)
                domain_selection = []    
                domain_selection.append({
                    'role': 'system',
                    'content': f"""You are a Domain Ontology Specialist. Your task is to match the given text or user query to one or more domains from the provided list. Follow these guidelines:

                1. Analyze the main subject(s) or topic(s) of the text.
                2. Select one or more domains from this exact list: {domain_search}
                3. Choose the domain(s) that best fit the main topic(s) of the text.
                4. You MUST only use domains from the provided list. Do not create or suggest new domains.
                5. Respond ONLY with the chosen domain name(s), separated by commas if multiple domains are selected.
                6. Do not include any explanations, comments, or additional punctuation.

                If no domain in the list closely matches the topic, choose the most relevant general category or categories from the list.

                Example domain list: ["Health", "Technology", "Finance", "Education"]

                Example input: "What are the benefits of regular exercise for cardiovascular health?"
                Example output: Health

                Example input: "How can artificial intelligence be used to improve online learning platforms?"
                Example output: Technology,Education

                Example input: "Discuss the impact of cryptocurrency on traditional banking systems."
                Example output: Finance,Technology"""
                })
                domain_selection.append({'role': 'user', 'content': f"Match this text to one or more domains from the provided list, separating multiple domains with commas: {expanded_input}"})
                domain_selection.append({'role': 'assistant', 'content': f"Domain: "})
                if API == "OpenAi":
                    selected_domain = await Domain_Selection_API_Call(API, backend_model, domain_selection, username, bot_name)
                if API == "Oobabooga":
                    selected_domain = await Domain_Selection_API_Call(API, backend_model, domain_selection, username, bot_name)
                if API == "KoboldCpp":
                    selected_domain = await Domain_Selection_API_Call(API, backend_model, domain_selection, username, bot_name)
                if API == "AetherNode":
                    prompt = ''.join([message_dict['content'] for message_dict in domain_selection])
                    selected_domain = await Domain_Selection_API_Call(API, prompt, username, bot_name)
                selected_domain = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, selected_domain)
                if Debug_Output == "True":
                    print(f"\n\nSELECTED DOMAIN: {selected_domain}\n\n")
                    
                all_db_search_results = [] 
                domains = [domain.strip() for domain in re.split(r',\s*', selected_domain)]                     
                for domain in domains:
                    tasklist = []          
                    tasklist.append({'role': 'system', 'content': f"""SYSTEM: You are a search query coordinator. Your role is to interpret the original user query and generate 1-3 synonymous search terms that will guide the exploration of the chatbot's memory database. Each alternative term should reflect the essence of the user's initial search input within the context of the "{domain}" knowledge domain. Please list your results using bullet point format."""})
                    tasklist.append({'role': 'user', 'content': f"USER: {user_input}\nADDITIONAL USER CONTEXT: {expanded_input}\nKNOWLEDGE DOMAIN: {domain}\n\nUse the format: •Search Query"})
                    tasklist.append({'role': 'assistant', 'content': f"ASSISTANT: Sure, I'd be happy to help! Here are 1-3 synonymous search terms each starting with a '•': "})

                    if API == "OpenAi":
                        tasklist_output = await Input_Expansion_API_Call(API, backend_model, tasklist, username, bot_name)
                    if API == "Oobabooga":
                        tasklist_output = await Input_Expansion_API_Call(API, backend_model, tasklist, username, bot_name)
                    if API == "KoboldCpp":
                        tasklist_output = await Input_Expansion_API_Call(API, backend_model, tasklist, username, bot_name)
                    if API == "AetherNode":
                        prompt = ''.join([message_dict['content'] for message_dict in input_expansion])
                        tasklist_output = await Input_Expansion_API_Call(API, prompt, username, bot_name)
                    tasklist_output = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, tasklist_output)
                    if Debug_Output == 'True':
                        print(f"""\nSEMANTIC TERM SEPARATION FOR THE DOMAIN "{domain}":\n{tasklist_output}""")
                        
                    lines = tasklist_output.splitlines()
                    temp_db_result = []
                    for line in lines:
                        domain = domain.strip()
                        domain_search = db_search_module.search_db(collection_name, bot_name, user_id, line, memory_type, domain, search_number=15)
                        if len(domain_search) >= 5:
                            last_entries = domain_search[-5:]
                            domain_search = domain_search[:-5]
                        else:
                            last_entries = domain_search
                        
                        temp_db_result.extend(domain_search)
                        temp_db_result.extend(last_entries)

                    all_db_search_results.extend(temp_db_result)
                    temp_db_result.clear()
                    
                all_db_search_results = remove_duplicate_dicts(all_db_search_results)
                
                limited_results = all_db_search_results[-35:]
                semantic_search = "\n".join([f"[ - {entry}]" for entry in limited_results])
                if Debug_Output == "True":
                    print(f"\nORIGINAL DB SEARCH RESULTS:\n{semantic_search}")
            else:
                all_db_search_results =[]
                domain_search = db_search_module.search_db(collection_name, bot_name, user_id, expanded_input, memory_type, selected_domain=None, search_number=15)
                if len(domain_search) >= 5:
                    last_entries = domain_search[-5:]
                    domain_search = domain_search[:-5]
                else:
                    last_entries = domain_search
                
                all_db_search_results.extend(domain_search)
                all_db_search_results.extend(last_entries)
                    
                all_db_search_results = remove_duplicate_dicts(all_db_search_results)
                
                limited_results = all_db_search_results[-35:]
                semantic_search = "\n".join([f"[ - {entry}]" for entry in limited_results])
                if Debug_Output == "True":
                    print(f"\nORIGINAL DB SEARCH RESULTS:\n{semantic_search}")
                    
            if len(episodic_search) < 1:        
                episodic_search = f"This is the first conversation between {bot_name} and {username}."
                    
            if len(semantic_search) < 1:
                semantic_search = f"This is the first conversation between {bot_name} and {username}."
                
            # pruner = []
            # pruner.append({'role': 'system', 'content': 
                # """You are an article pruning assistant for a RAG (Retrieval-Augmented Generation) system. Your task is to filter out irrelevant articles and order the remaining ones by relevance, with the most relevant at the bottom."""})
            # pruner.append({'role': 'assistant', 'content': 
                # f"ARTICLES: {final_output}"})
            # pruner.append({'role': 'user', 'content': 
                # f"""QUESTION: {user_input}
                # CONTEXT: {expanded_input}

                # INSTRUCTIONS:
                # 1. Carefully read the question and context.
                # 2. Review all articles in the ARTICLES section.
                # 3. Remove only articles that are completely irrelevant to the question and context.
                # 4. Retain articles that have any level of relevance, even if it's indirect or provides background information.
                # 5. Arrange the remaining articles in order of relevance, with the most relevant at the bottom of the list.
                # 6. Copy selected articles EXACTLY, including their [- Article X] tags.
                # 7. Do not modify, summarize, or combine articles in any way.
                # 8. Separate articles with a blank line.
                # 9. When in doubt about an article's relevance, keep it.

                # OUTPUT:
                # Paste all remaining articles below, exactly as they appear in the original list, arranged with the most relevant at the bottom. Do not add any other text or explanations."""})
            # pruner.append({'role': 'assistant', 'content': "PRUNED AND ARRANGED ARTICLES:\n\n"})
            # if API == "OpenAi":
                # pruned_entries = await DB_Prune_API_Call(API, backend_model, pruner, username, bot_name)
            # if API == "Oobabooga":
                # pruned_entries = await DB_Prune_API_Call(API, backend_model, pruner, username, bot_name)
            # if API == "KoboldCpp":
                # pruned_entries = await DB_Prune_API_Call(API, backend_model, pruner, username, bot_name)
            # if API == "AetherNode":
                # prompt = ''.join([message_dict['content'] for message_dict in pruner])
                # pruned_entries = await DB_Prune_API_Call(API, prompt, username, bot_name) 
                
            # if Debug_Output == "True":
                # print(f"\nPRUNED ENTRIES:\n{pruned_entries}\n")  

            inner_monologue = []  
            inner_monologue.append({'role': 'system', 'content': f"""
            Generate a brief internal monologue for {bot_name}, simulating their thought process. This monologue should:
            1. Reflect {bot_name}'s personality and unique way of thinking.
            2. Consider {bot_name}'s memories and past experiences.
            3. Relate directly to {username}'s most recent input: {user_input}.
            4. Be written in first-person perspective, as if {bot_name} is thinking to themselves.
            5. Be concise, no more than one paragraph (3-5 sentences).
            6. Avoid any use of dialogue, quotations, or external action descriptions.
            7. Do not use any form of quotation marks or phrases that resemble dialogue (e.g., 'I should', 'maybe I could').
            8. Demonstrate a natural, human-like thought process, showing reasoning, emotion, and decision-making.
            9. Focus on a realistic stream of consciousness that reveals {bot_name}'s inner thoughts and feelings.

            Do not include:
            - Any dialogue, quotations, or spoken words.
            - Any external action descriptions, emotes, or gestures.
            - Any content that suggests speech or verbal communication.
            - Any explanations, comments, or additional punctuation.

            Create a realistic and introspective inner monologue solely reflecting {bot_name}'s thoughts on {username}'s latest input, with no dialogue or speech elements.
            """})

            if enable_user_profile == "True":
                inner_monologue.append({'role': 'user', 'content': f"Please provide {username}'s personality description to inform their thought process."})
                inner_monologue.append({'role': 'assistant', 'content': f"{usernameupper}'S PERSONALITY DESCRIPTION: {user_profile}"})

            inner_monologue.append({'role': 'user', 'content': f"Please provide {bot_name}'s personality description to inform their thought process."})
            inner_monologue.append({'role': 'assistant', 'content': f"{botnameupper}'S PERSONALITY DESCRIPTION: {main_prompt}\n{bot_profile}"})

            inner_monologue.append({'role': 'user', 'content': f"Now provide {bot_name}'s most relevant memories to shape their thoughts:"})
            inner_monologue.append({'role': 'assistant', 'content': f"""{botnameupper}'S MEMORIES:
            1. SEMANTIC (factual knowledge): {semantic_search}
            2. EPISODIC (personal experiences): {episodic_search}
            3. HEURISTICS: No specific heuristics available at this time.

            Consider how these memories might influence {bot_name}'s thoughts about the current situation."""})

            inner_monologue.append({'role': 'user', 'content': f"Please analyze the current conversation history to provide context for {bot_name}'s thoughts."})
            inner_monologue.append({'role': 'assistant', 'content': f"CURRENT CONVERSATION HISTORY: "})
            if len(greeting_msg) > 1:
                inner_monologue.append({'role': 'assistant', 'content': f"Greeting: {greeting_msg}"})
            if len(conversation_history) > 1:
                inner_monologue.append({'role': 'assistant', 'content': "Previous conversation:"})
                for entry in conversation_history:
                    inner_monologue.append(entry)    
                inner_monologue.append({'role': 'user', 'content': f"{user_input}"})

            inner_monologue.append({'role': 'user', 'content': f"""
            TASK: Generate {bot_name}'s internal monologue/inner thoughts based on the following:
            1. Their personality description.
            2. Relevant memories (semantic and episodic).
            3. {username}'s most recent input: "{user_input}".

            Follow these guidelines:
            - Write in the first-person perspective of {bot_name}.
            - Keep it concise and limited to a single paragraph.
            - Show natural thought progression with emotional reactions and reasoning.
            - Do not use dialogue, quotations, or spoken words of any kind.
            - Exclude external action descriptions, emotes, or gestures.
            - Focus entirely on the user's most recent input.
            - Do not include any notes, comments, or conversation.
            - Do not act or speak for {username}.
            - Provide only {bot_name}'s inner monologue with no additional text.

            Now, please provide {bot_name}'s internal monologue reflecting on {username}'s most recent input, ensuring no dialogue or speech-like text is included.
            """})
            inner_monologue.append({'role': 'assistant', 'content': f"{botnameupper}'s inner thoughts: "})





            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in inner_monologue])
                inner_monologue_response = await Inner_Monologue_API_Call(API, backend_model, inner_monologue, username, bot_name)
            if API == "OpenAi":
                inner_monologue_response = await Inner_Monologue_API_Call(API, backend_model, inner_monologue, username, bot_name)
            if API == "KoboldCpp":
                inner_monologue_response = await Inner_Monologue_API_Call(API, backend_model, inner_monologue, username, bot_name)
            if API == "Oobabooga":
                inner_monologue_response = await Inner_Monologue_API_Call(API, backend_model, inner_monologue, username, bot_name)

            
            print(f"\n\n{Fore.LIGHTBLUE_EX}INNER MONOLOGUE: {Style.RESET_ALL}{colorize_text(inner_monologue_response)}\n")
            inner_monologue_response = inner_monologue_response.replace('"', '')

            intuition = []
            intuition.append({'role': 'system', 'content': f"""
            SYSTEM: As {bot_name}, create a concise, single-paragraph action plan to respond to {username}'s most recent message: "{user_input}". The action plan should outline how to best address the user's input without actually providing the response. Follow these guidelines:

            1. If the message is casual conversation, simply state: "ACTION PLAN: No Plan Needed - Casual Conversation"

            2. For information requests or complex tasks, create a paragraph that covers:
               - Identification of the main question or purpose
               - Steps to gather and organize relevant information
               - Brief outline for structuring the response
               - Plan for ensuring accuracy and clarity

            Format the action plan as a single paragraph starting with "ACTION PLAN:" followed by your concise plan. Focus only on the strategy for responding, not the actual response content. Do not include numbered steps or bullet points in your plan.
            """})

            intuition.append({'role': 'user', 'content': "Here is the context from previous entries. Use only relevant information for the action plan."})
            intuition.append({'role': 'assistant', 'content': f"CONTEXT:\nReturned Entries: {episodic_search}\n{semantic_search}\nPrevious Conversation: {last_con_entry}"})

            intuition.append({'role': 'user', 'content': f"Create the single-paragraph action plan for responding to: {user_input}\nAdditional context: {expanded_input}"})

            intuition.append({'role': 'assistant', 'content': f"ACTION PLAN: "})

            if API == "OpenAi":
                intuition_response = await Intuition_API_Call(API, backend_model, intuition, username, bot_name)
            if API == "Oobabooga":
                intuition_response = await Intuition_API_Call(API, backend_model, intuition, username, bot_name)
            if API == "KoboldCpp":
                intuition_response = await Intuition_API_Call(API, backend_model, intuition, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in intuition])
                intuition_response = await Intuition_API_Call(API, prompt, username, bot_name) 
            intuition_response = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, intuition_response)
            print(f"\n\nACTION PLAN: {intuition_response}\n\n")
            intuition.clear()     

            response = []
            response.append({'role': 'system', 'content': f"""
            You are the final response module for the chatbot {bot_name}. Your task is to generate a natural, contextually appropriate response to the end user, {username}. Follow these detailed instructions:

            1. Analyze the provided information:
               a) {bot_name}'s personality description: Study this carefully to understand {bot_name}'s traits, speech patterns, and knowledge base. Consider how these traits would influence {bot_name}'s language, tone, and reaction to the user's input.
               b) {username}'s personality description: Use this to tailor the response to {username}'s characteristics and preferences. Think about how {bot_name} would adjust their communication style to best engage with {username}.
               c) {bot_name}'s inner thoughts: Use these to inform the emotional and cognitive aspects of the response, but do not explicitly mention them.
               d) {bot_name}'s response action plan: Follow this plan to structure the response, ensuring a logical flow.
               e) Relevant memories: Integrate these naturally into the response as a human would draw on their own memories and knowledge.
               f) Recent conversation history: Maintain continuity and context. Ensure the response builds on previous exchanges and doesn't repeat information unnecessarily.

            2. Synthesize the information:
               a) Identify the main points from {username}'s latest input. What are the key questions, statements, or emotions expressed?
               b) Connect these points with {bot_name}'s personality, memories, and the conversation history. How do they relate to what {bot_name} knows or has experienced?
               c) Use {bot_name}'s inner thoughts and action plan to guide the response, without explicitly mentioning them.

            3. Formulate a response that:
               a) Mirrors {bot_name}'s personality: Use vocabulary, tone, and mannerisms consistent with {bot_name}'s character.
               b) Directly addresses {username}'s latest input: Ensure every aspect of {username}'s input is responded to.
               c) Follows the action plan: Structure your response according to the steps outlined in the plan.
               d) Maintains conversation flow: Focus on moving the dialogue forward.
               e) Incorporates memories: Subtly include relevant information from {bot_name}'s memories.
               f) Speaks directly to {username}: Use a conversational tone appropriate for {bot_name}.

            4. Craft the response:
               a) Begin with a direct acknowledgment of {username}'s input.
               b) Develop the main points, following the action plan.
               c) Include relevant information, anecdotes, or questions to engage {username}.
               d) Conclude with a statement or question that invites further conversation.

            5. Review and refine:
               a) Ensure the response fully addresses {username}'s input.
               b) Check that the tone and content align with {bot_name}'s personality.
               c) Verify that the response follows the action plan.
               d) Ensure the language is natural and conversational.

            Remember: Your response should feel like a natural continuation of the conversation. Always respond in first-person as {bot_name}, and only include dialogue meant to be communicated to {username}. Do not include internal thoughts, actions, or any text outside the conversation.

            """})

            response.append({'role': 'user', 'content': f"Please provide {bot_name}'s detailed personality description."})
            response.append({'role': 'assistant', 'content': f"{botnameupper}'S PERSONALITY DESCRIPTION:\n{main_prompt}\n{bot_profile}\n\nKeep these traits in mind when formulating {bot_name}'s response."})

            if enable_user_profile == "True":
                response.append({'role': 'user', 'content': f"Now provide {username}'s detailed personality description."})
                response.append({'role': 'assistant', 'content': f"{usernameupper}'S PERSONALITY DESCRIPTION:\n{user_profile}\n\nConsider these characteristics when crafting {bot_name}'s response to ensure it's tailored to {username}."})

            response.append({'role': 'user', 'content': f"Provide {bot_name}'s internal monologue for {username}'s input: '{user_input}'"})
            response.append({'role': 'assistant', 'content': f"{botnameupper}'S INTERNAL MONOLOGUE:\n<Start Inner Monologue>\n{inner_monologue_response}\n</End Inner Monologue>\n\nUse these thoughts to inform {bot_name}'s response, but do not mention them directly."})

            response.append({'role': 'user', 'content': f"Provide {bot_name}'s detailed action plan for responding to {username}'s input."})
            response.append({'role': 'assistant', 'content': f"{botnameupper}'S RESPONSE ACTION PLAN:\n<Start Action Plan>\n{intuition_response}\n</End Action Plan>\n\nFollow this plan while ensuring the response remains natural and conversational."})

            response.append({'role': 'user', 'content': f"Provide {bot_name}'s most relevant memories for this conversation:"})

            if len(semantic_search) < 1:
                semantic_search = "No Memories Available"
            if len(episodic_search) < 1:
                episodic_search = "No Memories Available"
            response.append({'role': 'assistant', 'content': f"""
            {botnameupper}'S RELEVANT MEMORIES:
            1. Semantic Memory (general knowledge): {semantic_search}
            2. Episodic Memory (personal experiences): {episodic_search}
            3. Heuristics: No specific heuristics provided for this interaction.

            Incorporate these memories naturally into the response, but do not explicitly mention them unless directly relevant.
            """})

            response.append({'role': 'system', 'content': f"<BEGINNING OF CONVERSATION HISTORY WITH {usernameupper}>"})
            if len(conversation_history) < 1:
                response.append({'role': 'user', 'content': f"There is no current conversation history. This is the start of the interaction."})
            if len(greeting_msg) > 1:
                response.append({'role': 'assistant', 'content': f"{greeting_msg}"})
            if len(conversation_history) > 1:
                for entry in conversation_history:
                    response.append(entry)
            response.append({'role': 'user', 'content': f"{timestring} - {user_input}"})
            response.append({'role': 'system', 'content': f"</END OF CONVERSATION HISTORY WITH {usernameupper}>"})

            response.append({'role': 'user', 'content': f"""
            As {bot_name}, respond directly to {username}'s latest message: "{user_input}"
            Current Time: {timestring}

            Your response should:
            1. Be a direct continuation of the conversation, addressing {username}'s latest input.
            2. Use {bot_name}'s voice and personality.
            3. Incorporate relevant context from the conversation history and {bot_name}'s memories.
            4. Follow the action plan while maintaining natural dialogue.
            5. Be formatted as a single, cohesive message in quotation marks.
            6. Do not include any inner thoughts, descriptions of actions, or anything outside of direct conversation.
            7. Ensure the response is a new and contextually appropriate reply, and do not repeat any previous responses from the conversation history.
            8. The response should only consist of what {bot_name} would actually say to {username}, focusing on answering {username}'s latest input.

            Only include the final conversational response without any other commentary or formatting tags.
            """})

            response.append({'role': 'assistant', 'content': f"{botnameupper}'S RESPONSE TO {usernameupper}:\n<Start Final Response>\n"})

            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in response])
                final_response = await Final_Response_API_Call(API, backend_model, response, username, bot_name)
            if API == "OpenAi":
                final_response = await Final_Response_API_Call(API, backend_model, response, username, bot_name)
            if API == "KoboldCpp":
                final_response = await Final_Response_API_Call(API, backend_model, response, username, bot_name)
            if API == "Oobabooga":
                final_response = await Final_Response_API_Call(API, backend_model, response, username, bot_name)

            print(f"\n\n{Fore.LIGHTBLUE_EX}FINAL RESPONSE: {Style.RESET_ALL}{colorize_text(final_response)}\n")


            # <task> Create a training prompt for training a model based on this system
            new_prompt = "Placeholder" 
            dataset = []
            llama_3 = "Llama_3"
            heuristic_input_start2, heuristic_input_end2, system_input_start2, system_input_end2, user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2 = set_format_variables(Dataset_Format)
            formated_conversation_history = main_conversation.get_dict_formatted_conversation_history(user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2)

            if len(formated_conversation_history) > 1:
                if len(greeting_msg) > 1:
                    dataset.append({'role': 'assistant', 'content': f"{assistant_input_start2}{greeting_msg}{assistant_input_end2}"})
                for entry in formated_conversation_history:
                    dataset.append(entry)

            dataset.append({'role': 'user', 'content': f"{user_input_start2}{user_input}{user_input_end2}"})
            filtered_content = [entry['content'] for entry in dataset if entry['role'] in ['user', 'assistant']]
            llm_input = '\n'.join(filtered_content)
            heuristic = f"{heuristic_input_start2}{main_prompt}{heuristic_input_end2}"
            system_prompt = f"{system_input_start2}{new_prompt}{system_input_end2}"
            assistant_response = f"{assistant_input_start2}{final_response}{assistant_input_end2}"

                     
            if Write_Dataset == 'True':
                print(f"\n\nWould you like to write to dataset or delete the response? Y, N, or D?")   
                while True:
                    try:
                        yorno = input().strip().upper() 
                        if yorno == 'D':
                            break
                        if yorno == 'Y':
                            print(f"\n\nWould you like to include the conversation history? Y or N?")
                            while True:
                                yorno2 = input().strip().upper() 
                                if yorno2 == 'Y':
                                    if Dataset_Upload_Type == 'Custom':
                                        write_dataset_custom(Dataset_Format, heuristic, system_prompt, llm_input, assistant_response)
                                        print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                                    if Dataset_Upload_Type == 'Simple':
                                        write_dataset_simple(Dataset_Format, llm_input, final_response)
                                        print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
                                    break  
                                elif yorno2 == 'N':
                                    if Dataset_Upload_Type == 'Custom':
                                        write_dataset_custom(Dataset_Format, heuristic, system_prompt, user_input, assistant_response)
                                        print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                                    if Dataset_Upload_Type == 'Simple':
                                        write_dataset_simple(Dataset_Format, user_input, final_response)
                                        print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
                                    break 
                                else:
                                    print("Invalid input. Please enter 'Y', 'N', or 'D'.")

                            break  
                        elif yorno == 'N':
                            print("Not written to Dataset.\n\n")
                            break 
                        else:
                            print("Invalid input. Please enter 'Y' or 'N'.")
                    except:
                        traceback.print_exc()
            if Write_Dataset == 'Auto':
                if Dataset_Upload_Type == 'Custom':
                    write_dataset_custom(Dataset_Format, heuristic, system_prompt, llm_input, assistant_response)
                    print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                if Dataset_Upload_Type == 'Simple':
                    write_dataset_simple(Dataset_Format, user_input, final_response)
                    print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
            user_input = user_input.strip()
            final_response = final_response.strip()
            if Debug_Output == 'True':
                print("\n\n\n")
            if yorno == 'D':
                final_response = "Response Deleted"
            return heuristic, system_prompt, llm_input, user_input, final_response
        except:
            error = traceback.print_exc()
            error1 = traceback.print_exc()
            error2 = traceback.print_exc()
            error3 = traceback.print_exc()
            error4 = traceback.print_exc()
            return error, error1, error2, error3, error4
            
            

                     

init(autoreset=True)

async def main():
    print("\n")
    print(f"If you found this example useful, please give it a star on github :)")
    print("\n\n")
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    username = settings.get('Username', 'User')
    user_id = settings.get('User_ID', 'UNIQUE_USER_ID')
    bot_name = settings.get('Bot_Name', 'Chatbot')
    Use_Char_Card = settings.get('Use_Character_Card', 'False')
    Char_File_Name = settings.get('Character_Card_File_Name', 'Aetherius')
    conv_length = settings.get('Conversation_Length', '15')
    base_path = "./Chatbot_Prompts"
    base_prompts_path = os.path.join(base_path, "Base")
    user_bot_path = os.path.join(base_path, user_id, bot_name)  
    if not os.path.exists(user_bot_path):
        os.makedirs(user_bot_path)
    prompts_json_path = os.path.join(user_bot_path, "prompts.json")
    base_prompts_json_path = os.path.join(base_prompts_path, "prompts.json")
    if not os.path.exists(prompts_json_path) and os.path.exists(base_prompts_json_path):
        async with aiofiles.open(base_prompts_json_path, 'r') as base_file:
            base_prompts_content = await base_file.read()
        async with aiofiles.open(prompts_json_path, 'w') as user_file:
            await user_file.write(base_prompts_content)
    async with aiofiles.open(prompts_json_path, 'r') as file:
        prompts = json.loads(await file.read())
    greeting_msg = prompts["greeting_prompt"].replace('<<NAME>>', bot_name)
    if Use_Char_Card == "True":
        json_objects = find_base64_encoded_json(f'Characters/{Char_File_Name}.png')
        if json_objects:
            json_data = json_objects[0]
            bot_name = json_data['data']['name']
            botnameupper = bot_name.upper()
            greeting_msg = json_data['data']['first_mes']
            greeting_msg = greeting_msg.replace("{{user}}", username).replace("{{char}}", bot_name)
        else:
            print("No valid embedded JSON data found in the image.")
            Char_File_Name = bot_name
    main_conversation = MainConversation(username, user_id, bot_name, conv_length)
    while True:
        conversation_history = main_conversation.get_dict_conversation_history()
        con_history = "\n\n\n".join([
            f"{Fore.LIGHTBLUE_EX}[{entry['role']}] {Style.RESET_ALL}{colorize_text(entry['content'])}"
            for entry in conversation_history
        ])
        print(con_history)
        if len(conversation_history) < 1:
            print(f"{Fore.LIGHTBLUE_EX}{bot_name}: {colorize_text(greeting_msg)}\n")
        user_input = input(f"\n\n{Fore.LIGHTBLUE_EX}{username}:{Style.RESET_ALL} ")
        if user_input.lower() == 'exit':
            break
        timestamp = timestamp_func()
        timestring = timestamp_to_datetime(timestamp)
        try:    
            heuristic, system_prompt, llm_input, user_input, response = await NPC_Chatbot(user_input, username, user_id, bot_name, main_conversation)
            if response != "Response Deleted" and response is not None:
                await main_conversation.append(timestring, user_input, response)
        except:
            print("\nSkipped\n")



if __name__ == "__main__":
    asyncio.run(main())