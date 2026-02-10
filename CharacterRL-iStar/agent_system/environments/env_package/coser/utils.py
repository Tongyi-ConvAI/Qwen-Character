import copy
import io
import json
import logging
import os
import pdb
import pickle
import random
import re
import time
from typing import Dict, List

import __main__
import openai
import requests
import tiktoken


def setup_logger(name, log_file, level=logging.INFO, quiet=False):
	logger = logging.getLogger(name)
	logger.setLevel(level)

	if logger.hasHandlers():
		logger.handlers.clear()


	file_handler = logging.FileHandler(log_file, encoding='utf-8')
	file_handler.setLevel(level)
	file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler.setFormatter(file_formatter)
	logger.addHandler(file_handler)


	if not quiet:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
		console_handler.setFormatter(console_formatter)
		logger.addHandler(console_handler)

	return logger

logger = setup_logger(__name__, f'{__file__.split(".")[0]}.log', level=logging.INFO, quiet=False)

cache_path = 'cache.pkl'
cache_sign = True
cache = None
reload_cache = False

def cached(func):
	def wrapper(*args, **kwargs):		
		key = ( func.__name__, str(args), str(kwargs.items()))
		
		global cache
		global reload_cache

		if reload_cache:
			cache = None # to reload
			reload_cache = False
		
		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				try:
					cache = pickle.load(open(cache_path, 'rb'))  
				except Exception as e:
					# print cache_path and throw error
					logger.error(f'Error loading cache from {cache_path}, set cache to empty dict')
					cache = {}

		if (cache_sign and key in cache) and not (cache[key] is None or cache[key] == ''):
			return cache[key]
		else:
			result = func(*args, **kwargs)
			if result != None:
				cache[key] = result
				pickle.dump(cache, open(cache_path, 'wb'))
				#safe_pickle_dump(cache, cache_path)

			return result

	return wrapper

def request_model_api(messages, model, max_tokens=8196, temperature=0.7):
	# set your url and token here
	url = ""
	token = ""
	headers = {
		'Authorization': f'Bearer {token}',
		'Content-Type': 'application/json'
	}
	data = {
	'model': model,
	'messages': messages,
	'max_tokens': max_tokens,
	'temperature': temperature
	}
	try:
		response = requests.post(url, headers=headers, json=data, timeout=120)
		response.raise_for_status() # 检查HTTP错误
		return response.json() 
	except requests.exceptions.RequestException as e:
		print(f"请求模型 {model} 时发生网络错误: {e}")
		return None
	except Exception as e:
		print(f"处理模型 {model} 请求时发生未知错误: {e}")
		return None

@cached
def get_response(model, messages, max_tokens=8196, nth_generation=0):
	# if messages is str
	if isinstance(messages, str):
		messages = [{"role": "user", "content": messages}]
		
	# correct 'system' to 'user'
	if model.startswith('claude') and messages and messages[0]['role'] == 'system': messages[0]['role'] = 'user'	

	# merge adjacent user messages
	merged_messages = []
	for message in messages:
		if message['role'] == 'user' and merged_messages and merged_messages[-1]['role'] == 'user':
			merged_messages[-1]['content'] += message['content']
		else:
			merged_messages.append(copy.deepcopy(message))

	messages = merged_messages
	response = request_model_api(messages, model, max_tokens=8196, temperature=0 if nth_generation == 0 else 1)
	if response and 'output' in response:
		response = response['output']

	return response['choices'][0]['message']['content']


def extract_json(text, **kwargs):
	def _fix_json(json_response):
		prompt = f'''I will provide you with a JSON string that contains errors, making it unparseable by `json.loads`. The most common issue is the presence of unescaped double quotes inside strings. Your task is to output the corrected JSON string. The JSON string to be corrected is:
{json_response}
'''

		response = get_response(model=kwargs['model'], messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response

	def _extract_json(text):
		# Use regular expressions to find all content within curly braces
		orig_text = text

		text = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group().replace('\n', r'\\n'), text) 
		
		#json_objects = re.findall(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)

		def parse_json_safely(text):
			try:
				result = json.loads(text)
				return result
			except json.JSONDecodeError:
				results = []
				start = 0
				while start < len(text):
					try:
						obj, end = json.JSONDecoder().raw_decode(text[start:])
						results.append(obj)
						start += end
					except json.JSONDecodeError:
						start += 1
				
				if results:
					longest_json = max(results, key=lambda x: len(json.dumps(x)))
					return longest_json
				else:
					return None
		
		extracted_json = parse_json_safely(text)
		
		if extracted_json:
			return extracted_json
		else:
			logger.error('Error parsing response: ', orig_text)
			return None

	res = _extract_json(text)

	if res:
		return res
	else:
		return _extract_json(_fix_json(text))

def get_response_json(model, messages: List[Dict], post_processing_funcs: List, max_retry: int = 5):
        """
        Get and process a response from an LLM with retries and error handling.
        Follows the same pattern as get_response_json in utils.py.
        
        Args:
            messages: List of message dicts for the LLM
            post_processing_funcs: List of functions to process the LLM response
            max_retry: Max number of retry attempts (default 5)
            
        Returns:
            dict: Processed JSON response from the LLM, or None if parsing fails
        """
        nth_generation = 0
        
        while nth_generation <= max_retry:
            response_text = get_response(model, messages, nth_generation=nth_generation)
            
            if response_text is None:
                nth_generation += 1
                continue
            
            # Apply post-processing functions
            response = response_text
            for post_processing_func in post_processing_funcs:
                response = post_processing_func(response, model=model)
            
            if response:
                return response
            else:
                nth_generation += 1
        
        return None


def remove_inner_thoughts(dialogue: str) -> str:
	cleaned_dialogue = re.sub(r'\[.*?\]', '', dialogue)
	cleaned_dialogue = '\n'.join(line.strip() for line in cleaned_dialogue.split('\n'))
	cleaned_dialogue = re.sub(r'\n+', '\n', cleaned_dialogue)
	
	return cleaned_dialogue.strip()  


if __name__ == '__main__':
	messages = [{"role": "system", "content": "How are you today?"}]
	model = "gpt-5-2025-08-07"
	# model = "gpt-4o-2024-08-06"
	# model = 'gemini-2.5-pro'

	print(get_response(model, messages))
		
