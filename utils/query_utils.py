import time
import os
import json
from typing import *
from tqdm import tqdm
import signal
from openai import OpenAI
from together import Together

def query_model(
        client: OpenAI|Together,
        model: str,
        input_prompt: str,
        system_prompt: str = None,
        chat: bool = True,
        logprobs: bool = False,
        echo: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ) -> Dict:
    client_api = client.__class__.__name__
    # Set up the call configuration
    call_config = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'logprobs': int(logprobs) if client_api == 'Together' else logprobs # Convert to binary for Together API
    }

    if chat:
        # Chat API requires a different configuration
        messages = [{"role": "user", "content": input_prompt}]
        messages = [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages # Add the system prompt if provided
        call_config.update({'messages': messages})
    else:
        assert not system_prompt, "System prompt is only available in chat mode. Consider adding it to the input prompt."
        call_config.update({'prompt': input_prompt})

    if echo:
        assert logprobs, "Echo requires logprobs to be enabled"
        call_config.update({'echo': 'true'})
    
    # Call the model
    response = client.chat.completions.create(**call_config) if chat else client.completions.create(**call_config)
    
    # Extract the data
    response = response.choices[0]
    data = {'message': response.message.content if chat else response.text}
    
    if logprobs:
        if client_api == 'Together':
            data.update({
                'tokens': response.logprobs.tokens,
                'logprobs': response.logprobs.token_logprobs
            })
        elif client_api == 'OpenAI':
            data.update({
                'tokens': [token.token for token in response.logprobs.content],
                'logprobs': [token.logprob for token in response.logprobs.content]
            })

    if echo:
        # Prepend the echo to the message
        data['tokens'] = response.prompt[0].logprobs.tokens + data['tokens']
        data['logprobs'] = response.prompt[0].logprobs.token_logprobs + data['logprobs']

    data = {k: v for k, v in data.items() if v is not None} # Remove None values
    return data

def run_experiment(
        input_prompts: List[str],
        models: List[str],
        results_path: str,
        system_prompt: str = None,
        query_timeout: int = 60,  # Timeout in seconds for each model query
        rate_limit: float = 0.02,  # Time to sleep between queries
        **kwargs
    ) -> Dict:
    # Define a handler for the timeout
    def handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, handler)

    # Initialize Client APIs
    together_client = Together()
    openai_client = OpenAI()

    # Get available models ids
    together_models = [model.id for model in together_client.models.list()]
    openai_models = [model.id for model in openai_client.models.list()]
    
    # Check if the models are available
    for model in models:
        if model not in together_models and model not in openai_models:
            raise ValueError(f"Model {model} not found in available models")

    # Initialize dictionary to store the data
    if not os.path.exists(results_path):
        json.dump({'data': {}}, open(results_path, 'w'))  # Create an empty file
    data = json.load(open(results_path, 'r'))
    
    # Remove models that have already been queried
    models = [model for model in models if model not in data['data'].keys()]        
    if not models:
        print('All models have already been queried')
        return data

    # Loop over models
    for idx, model in enumerate(models):
        print(f'_____ Model: {model} ({idx+1}/{len(models)}) _____')

        # Get the client for the model
        client = together_client if model in together_models else openai_client

        # Initialize a list to store the model responses
        model_responses = []

        # Input all the items to the model
        for input_prompt in tqdm(input_prompts):
            try:
                signal.alarm(query_timeout)  # Start the timeout clock
                response = query_model(client, model, input_prompt, system_prompt, **kwargs)
                model_responses.append(response)
                signal.alarm(0)  # Reset the alarm once the query completes
                time.sleep(rate_limit) # Sleep for a bit to avoid rate limiting
            except TimeoutError:
                print(f"Query for {model} timed out!")
                break
            except Exception as e:
                print(f"Calling {model} failed: {str(e)}")
                break
        
        # Check if all the items have been queried
        if len(model_responses) == len(input_prompts):
            data['data'][model] = model_responses
            # Save intermediate results
            json.dump(data, open(results_path, 'w'), indent=4)

    return data