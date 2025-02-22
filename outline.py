from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os

# Load environment variables
load_dotenv()

# Get API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    temperature=1,
)

# Define the Output Schema
class PlotOutline(BaseModel):
    outline: str = Field(description="Generated plot outline")

# Create JSON Output Parser
parser = JsonOutputParser(pydantic_object=PlotOutline)

# Define Prompt Template
outline_prompt_template = """
TASK: Generate a basic plot outline for a story based on the following premise. Provide 3-5 plot points in chronological order.

PREMISE: {userPremise}
GENRE: {userGenre}

The output must be in JSON format as follows:
```json
{{
    "outline": [
        "Plot point 1",
        "Plot point 2",
        "Plot point 3"
    ]
}}
"""

# Create Prompt
outline_prompt = PromptTemplate(
    input_variables=["userPremise", "userGenre"],
    template=outline_prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)



# Create Chain
outline_chain = outline_prompt | llm | parser

# Function to Generate Plot Outline
def generate_plot_outline(user_premise, user_genre):
    """
    Generates a basic plot outline for a story based on the provided premise and genre.

    Args:
        user_premise (str): The main idea or scenario of the story.
        user_genre (str): The genre of the story (e.g., Fantasy, Romance, Thriller).

    Returns:
        str: The generated plot outline in JSON format.
    """
    input_data = {
        "userPremise": user_premise,
        "userGenre": user_genre
    }
    
    # Invoke the chain and get the response
    response = outline_chain.invoke(input_data)
    return response
