import requests

def generate_llama_response(prompt):
    # The URL for the LMStudio local server (adjust the port if necessary)
    url = "http://localhost:1234/v1/chat/completions"
    
    # Payload to send to the LMStudio local server
    payload = {
        "model": "LM Studio Community/Meta-Llama-3-7B-Instruct",  # Update with your model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},  # System prompt
            {"role": "user", "content": prompt}  # User input prompt
        ],
        "temperature": 0.7,  # Adjust temperature for randomness
        "max_tokens": 512,   # Maximum tokens for the response
        "stream": False  # No streaming needed
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send POST request to LMStudio server
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse and return the generated content from the response
            return response.json()["choices"][0]["message"]["content"]
        else:
            # Return error message if the request failed
            return f"Error: {response.status_code}, {response.text}"
    
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"
