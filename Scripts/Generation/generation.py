from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to generate a response using google/flan-t5-small
def generate_response_with_flan_t5(query_text, combined_context):
    """
    Function to generate a response using Flan-T5 based on the query and the retrieved context.

    Args:
    - query_text (str): The query to ask.
    - combined_context (str): The retrieved context from the documents.

    Returns:
    - str: The generated response from Flan-T5.
    """
    # Load the Flan-T5 tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Combine the query and the retrieved context into a prompt
    input_text = f"Context: {combined_context}\n\nQuery: {query_text}"

    # Tokenize the input text for the model
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate a response using Flan-T5
    outputs = model.generate(inputs.input_ids, max_length=512, num_return_sequences=1,temperature=0.9)

    # Decode and return the generated response
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_response
