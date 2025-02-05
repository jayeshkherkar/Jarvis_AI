import cohere 
from rich import print 
from dotenv import dotenv_values

#Load environment variables from the .env files

env_vars = dotenv_values(".env")

# Retrieve API key
CohereAPIkey = env_vars.get("CohereAPIkey")    

# Create a cohere client using the provided API key
co = cohere.Client(api_key=CohereAPIkey)

# Define a list of recognized function keywords for task categorization.
funcs = ["exit","general","realtime","open","close","play","generate image","system","content","google search","youtube search","reminder"]

# Initialize an empty list to store user messages.
messages = []

# Define the preamble that guides the AI models on how to categorize queries.
preamble = """ """

# Define a chat history with predefined user -chatbot interactions for context.
ChatHistory = [
    
    {"role": "user","message":"how are you"},
    {"role": "Chatbot","message":"general how are you?"},
    {"role": "user","message":"do you like pizza?"},
    {"role": "Chatbot","message":"general do you like pizza?"},
    {"role": "user","message":"open chrome and tell me about mahatma gandhi."},
    {"role": "Chatbot","message":"open chrome, general tell me about mahatma gandhi."},
    {"role": "user","message":"open chrome and firefox"},
    {"role": "Chatbot","message":"open chrome, firefox"},
    {"role": "user","message":"what is today's data and by the way remind me that i have a dancing performance on 5th aug at 11pm"},
    {"role": "Chatbot","message":"general what is today's date, remainder 11:00pm 5th aug dancing performance"},
    {"role": "user","message":"chat with me."},
    {"role": "Chatbot","message":"general chat with me."},    
    
]

# Define the main function for decision-making on queries.
def FirstLayerDMM(prompt: str = "test"):
    
    # Add the user's query to the messages list.
    messages.append({"role":"user", "content":f"{prompt}"})
    
    # Create a streaming chat session with the Cohere model.
    stream = co.chat_stream(
        model='command-r-plus', #Specify the Cohere model to use.
        message=prompt, #Pass the user's query.
        temperature=0.7, #Set the creativity level of the model.
        chat_history=ChatHistory, # Provide the predefined chat history for context. 
        prompt_truncation='OFF', # Ensure the prompt is not truncated.
        connectors=[], # No additional connectors are used.
        preamble=preamble #Pass the detailed instruction preamble. 
    )
    
    # Initialize an empty string to store the generated response.
    response = ""
    
    # Iterate over agents in the stream and capture text generation events.
    for event in stream:
        if event.event_type == "text-generation":
            response += event.text # Append generated text to the response. 
            
    # Remove newline characters and split responses into individual tasks.
    response = response.replace("\n","")
    response = response.split(",")  
    
    # Strip leading and trailing whitespaces from each tesk
    response = [i.strip() for i in response]
    
    # Initialize an empty list to filter valid tasks.
    temp = []
    
    # Filter the tasks based on recognized functional keywords
    for task in response:
        for func in funcs:
            if task.startswith(func):
                temp.append(task)  # Add valid tasks to the filtered list of tasks.
                
    
    # Update the response with the filtered list of tasks.
    response = temp
    
    # If '(query)' is in the response, recursive call the function for further clarification.
    if "(query)" in response:
        newresponse = FirstLayerDMM(prompt=prompt)
        return newresponse
    else:
        return response   #Return the filtered respones.
    
    #Entry point for the script 
    if __name__== "__main__":
        # Continuously prompt the user for input and process it.
        while True:
            print(FirstLayerDMM(input(">>> ")))  # Print the categorized response.
    
         
    

