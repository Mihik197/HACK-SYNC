from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

# Initialize the client
client = ElevenLabs(
    api_key=ELEVEN_LABS_API_KEY
)

def play_story_from_json(json_input):
    """
    This function takes a JSON input containing a story, converts it to speech using the ElevenLabs API,
    and plays the audio directly.
    
    Example input: '{"story": "Once upon a time..."}'
    """
    # Parse the JSON input to get the story text
    data = json.loads(json_input)
    story_text = data.get('story')
    
    if not story_text:
        return json.dumps({"error": "No story provided"})
    
    # Call the API to convert the text to speech
    audio = client.text_to_speech.convert(
        text=story_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # You can change this to another voice if needed
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    
    # Play the audio directly
    play(audio)