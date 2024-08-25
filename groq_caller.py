import os
from groq import Groq
# Initialize the Groq client
client = Groq(
    api_key="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",
)

def speechtoText():
    # Specify the path to the audio file
    filename = "tem_output.wav" # Replace with your audio file!

    # Open the audio file
    with open(filename, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
            file=(filename, file.read()), # Required audio file
            model="distil-whisper-large-v3-en", # Required model to use for transcription
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            language="en",  # Optional
            temperature=0.0  # Optional
        )
    
    print(f"user input is: {transcription.text}")
    paper_content = ""
    contextual_content = ""
    # transcriped text
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a paper summarier that answer questions about the paper I passed in. \
                    Also, I will pass in contextual content to help you understand the paper. Then, you will answer my questions based on the paper.\
                    The paper content is as following {paper_content}, \
                    And the contextual content is as following {contextual_content}",
            },
            {
                "role": "user",
                "content": transcription.text,
            },
        ],
        model="llama-3.1-70b-versatile",
    )

    print(f"the output is {chat_completion.choices[0].message.content}")