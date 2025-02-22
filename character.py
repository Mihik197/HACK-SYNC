from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os

# Load environment variables
load_dotenv()

# Get GEMINI API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM with Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    temperature=1
)

# Define Output Schema
class CharacterProfile(BaseModel):
    name: str = Field(description="Name of the character")
    personality_traits: list = Field(description="3-5 key personality traits of the character")
    backstory: str = Field(description="Short backstory that aligns with the personality traits")

# Create Output Parser
parser = JsonOutputParser(pydantic_object=CharacterProfile)

# Define Character Prompt Template
character_prompt_template = """
    TASK: Generate a character profile based on the following description. 
    Provide a name, 3-5 key personality traits, and a short (1-2 sentence) backstory that aligns with the traits. 
    Make the character interesting and suitable for the specified genre. 
    Output the result in JSON format as follows:

    ```json
    {{
        "name": "Character's name",
        "personality_traits": ["trait1", "trait2", "trait3"],
        "backstory": "Short backstory text"
    }}
    ```

    CHARACTER DESCRIPTION: {userCharacterDescription}
    GENRE: {userGenre}
"""

# Create Prompt Template
character_prompt = PromptTemplate(
    input_variables=["userCharacterDescription", "userGenre"],
    template=character_prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Combine Prompt, LLM, and Parser into a Chain
character_chain = character_prompt | llm | parser

# Function to Generate Character Profile
def generate_character_profile(user_character_description: str, user_genre: str) -> dict:
    """
    Generates a character profile based on the provided description and genre.

    Args:
        user_character_description (str): Description of the character's role, appearance, or other traits.
        user_genre (str): The genre of the story (e.g., Fantasy, Romance, Thriller).

    Returns:
        dict: The generated character profile in JSON format.
    """
    # Prepare input data for the chain
    input_data = {
        "userCharacterDescription": user_character_description,
        "userGenre": user_genre
    }
    
    # Invoke the chain and get the response
    response = character_chain.invoke(input_data)
    
    # Return the generated character profile in JSON format
    return response
