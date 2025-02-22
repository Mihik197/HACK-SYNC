import base64
import json
import os
from together import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize the client
client = Together(
    api_key=TOGETHER_API_KEY
)

def generate_and_save_image(json_input):
    """
    Takes a JSON input containing a prompt, generates an image using the Together API,
    saves the image as a file, and returns a download link.
    
    Example input: '{"prompt": "A character continuously crying and in a very depressed mood"}'
    """
    # Parse the JSON input to get the prompt
    data = json.loads(json_input)
    prompt = data.get('prompt')
    
    if not prompt:
        return json.dumps({"error": "No prompt provided"})
    
    # Call the Together API to generate the image
    response = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-dev",
        width=1024,
        height=768,
        steps=28,
        n=1,
        response_format="b64_json",
    )
    
    # Extract the Base64 image data
    b64_json = response.data[0].b64_json
    
    # Decode the Base64 string to image data
    image_data = base64.b64decode(b64_json)
    
    # Save the image as a file
    image_filename = "generated_image.png"
    with open(image_filename, "wb") as file:
        file.write(image_data)
    
    # Get the absolute file path
    image_path = os.path.abspath(image_filename)
    
    # Return the download link (adjust the URL if serving from a web server)
    download_link = f"file://{image_path}"
    result = {
        "download_link": download_link
    }
    
    return json.dumps(result)