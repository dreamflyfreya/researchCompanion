import os
from groq import Groq
# Initialize the Groq client
import requests
import wave
CHUNK_SIZE = 1024

url = "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL"
def save_audio_to_wav(audio_data, output_filename, sample_width=2, channels=1, framerate=44100):
    with wave.open(output_filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(framerate)
        wav_file.writeframes(audio_data)

def texttospeech(text):
    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "sk_f1b3b36c47681228209be180fe972705797d67a37a90935a"
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        audio_data = response.content
        save_audio_to_wav(audio_data, 'final_output.wav')
        print("Audio saved successfully.")
    else:
        print(f"Error: {response.status_code}, {response.text}")

client = Groq(
    api_key="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",
)

def speechtoText():
    # Specify the path to the audio file
    filename = "output.wav" # Replace with your audio file!

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
    texttospeech(chat_completion.choices[0].message.content)

