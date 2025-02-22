# Importing Required Libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os

# Load Environment Variables
load_dotenv()

# Get GEMINI API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Language Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    temperature=1,
)

# Define Output Schema for JSON Response
class StoryChapter(BaseModel):
    chapter_text: str = Field(description="Full text of the generated chapter")

# Create JSON Output Parser
parser = JsonOutputParser(pydantic_object=StoryChapter)

# Prompt Template with JSON Output Specification
chapter_prompt_template = """
TASK: Write a chapter for a story based on the provided plot point, previous chapters (if any), and story context. 
Maintain a consistent style and tone throughout the chapter, aligning with the specified genre and any established themes. 
Develop the characters and worldbuilding organically within the chapter, according to the provided information.

Your output should be the length of an actual chapter in a novel, aiming for a substantial length between 2000 and 4000 words minimum, 
to allow for sufficient development and immersion, unless a different length is explicitly specified by the user in 'userStyle'. 
You are expected to write a high-quality and creative story chapter that is engaging and well-paced. Do not rush to conclude the chapter; 
let the narrative flow naturally and develop at a pace that feels appropriate for the story and the current plot point. 
Focus on creating a compelling and immersive reading experience.

PLOT POINT:
{plotPoint}

PREVIOUS CHAPTERS:
{previousChapters}

CHARACTERS:
{characterData}

WORLDBUILDING:
{worldbuildingData}

GENRE: {userGenre}

STYLE: {userStyle}  # User-specified style, can include length requests or stylistic preferences

The output must be in JSON format as follows:
```json
{{
    "chapter_text": "Full text of the generated chapter here"
}}
"""


chapter_prompt = PromptTemplate( input_variables=[ "plotPoint", "previousChapters", "characterData", "worldbuildingData", "userGenre", "userStyle" ], template=chapter_prompt_template )

chapter_chain = chapter_prompt | llm | parser

def generate_story_chapter( 
    plot_point: str, 
    previous_chapters: str, 
    character_data: str, 
    worldbuilding_data: str, 
    user_genre: str, 
    user_style: str 
) -> dict:
    """
    Generates a story chapter based on the provided plot point, previous chapters, character data, worldbuilding,
    genre, and style preferences.

    Args:
        plot_point (str): The main event or development for this chapter.
        previous_chapters (str): Text of the previous chapters for context.
        character_data (str): Description of characters involved in the story.
        worldbuilding_data (str): Information about the story's world, setting, and rules.
        user_genre (str): The genre of the story (e.g., Fantasy, Romance, Thriller).
        user_style (str): User-specified style, including length or stylistic preferences.

    Returns:
        dict: The generated story chapter in JSON format.
    """
    # Prepare input data for the chain
    input_data = {
        "plotPoint": plot_point,
        "previousChapters": previous_chapters,
        "characterData": character_data,
        "worldbuildingData": worldbuilding_data,
        "userGenre": user_genre,
        "userStyle": user_style
    }
    
    # Invoke the chain and get the response
    response = chapter_chain.invoke(input_data)
    
    # Return the generated chapter in JSON format
    return response
