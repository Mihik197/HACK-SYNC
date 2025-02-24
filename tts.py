from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
import json
# import base64
# from pygame import mixer
import tempfile
from elevenlabs import play

# Load environment variables
load_dotenv()
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

# Initialize the client
client = ElevenLabs(
    api_key=ELEVEN_LABS_API_KEY
)

def get_audio_base64_from_story(story_text):
    """
    This function takes a story text, converts it to speech using the ElevenLabs API,
    and returns the audio data as a base64-encoded string in JSON format.
    """
    if not story_text:
        return {"error": "No story provided"}
    
   
    audio = client.text_to_speech.convert(
            text=story_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Change this to another voice if needed
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # # Encode the audio bytes as a base64 string
        # audio_base64 = base64.b64encode(audio).decode('utf-8')
        
    play(audio)