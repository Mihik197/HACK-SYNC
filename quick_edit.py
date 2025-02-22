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
class EditedText(BaseModel):
    edited_text: str = Field(description="The edited version of the story text")

# Create Output Parser
parser = JsonOutputParser(pydantic_object=EditedText)

# Define Quick Edit Prompt Template
quick_edit_prompt_template = """
    TASK: Perform a quick edit on the provided text based on the user's specific request. 
    Consider the overall story context, characters, worldbuilding, and genre to ensure the edit is consistent and appropriate. 
    Fulfill the user's request concisely and effectively. Output the result in JSON format as follows:

    ```json
    {{
        "edited_text": "Your edited version here"
    }}
    ```

    USER REQUEST: {userRequest}

    CURRENT STORY TEXT:
    {documentText}

    CHARACTERS (for context):
    {characterData}

    WORLDBUILDING (for context):
    {worldbuildingData}

    GENRE (for context):
    {userGenre}
"""

# Create Prompt Template
quick_edit_prompt = PromptTemplate(
    input_variables=["userRequest", "documentText", "characterData", "worldbuildingData", "userGenre"],
    template=quick_edit_prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Combine Prompt, LLM, and Parser into a Chain
quick_edit_chain = quick_edit_prompt | llm | parser

# Function to Perform Quick Edit
def perform_quick_edit(user_request: str, document_text: str, character_data: str, worldbuilding_data: str, user_genre: str) -> dict:
    """
    Performs a quick edit on the provided story text based on the user's request and story context.

    Args:
        user_request (str): Specific editing request from the user.
        document_text (str): The current story text to be edited.
        character_data (str): Character information for context.
        worldbuilding_data (str): Worldbuilding details for consistency.
        user_genre (str): The genre of the story (e.g., Fantasy, Romance, Thriller).

    Returns:
        dict: The edited version of the story text in JSON format.
    """
    # Prepare input data for the chain
    input_data = {
        "userRequest": user_request,
        "documentText": document_text,
        "characterData": character_data,
        "worldbuildingData": worldbuilding_data,
        "userGenre": user_genre
    }
    
    # Invoke the chain and get the response
    response = quick_edit_chain.invoke(input_data)
    
    # Return the edited story text in JSON format
    return response
