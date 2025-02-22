from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from langchain_core.prompts import PromptTemplate



load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    temperature=1,
    
)
class rewrittenText(BaseModel):
    newText: str = Field(description="new_eddited_text")

parser = JsonOutputParser(pydantic_object=rewrittenText)

rewrite_prompt_template = {
    "system": """You are a helpful and creative writing assistant. You are collaborating with a writer on a story. Your current task is to REWRITE some sentences based on the writer's needs to improve the writing. Focus on clarity, style, and impact as requested by the writer. The response must be in JSON format as follows:

    ```json
    {{
        "rewritten_text": "Your rewritten version here"
    }}
    ```""",

    "shorter": """TASK: Rewrite the following text to make it significantly more concise and to the point. Remove any unnecessary words, phrases, or sentences. Aim for brevity without losing the core meaning.

    ORIGINAL TEXT:
    {selectedText}""",

    "longer": """TASK: Expand on the following text, adding more detail, description, and context. Maintain the original meaning but elaborate on it significantly. Enrich the text to provide a fuller and more immersive experience for the reader.

    ORIGINAL TEXT:
    {selectedText}""",

    "more_intense": """TASK: Rewrite the following text to dramatically increase the dramatic tension and emotional impact. Make it more gripping, suspenseful, and intense. Focus on heightening the emotional stakes and creating a stronger reaction in the reader.

    ORIGINAL TEXT:
    {selectedText}""",

    "more_descriptive": """TASK: Rewrite the following text to add rich sensory details (sight, sound, smell, taste, touch) and vivid imagery. Make it more descriptive and immersive by focusing on showing rather than telling. Engage the reader's senses and create a more vibrant scene.

    ORIGINAL TEXT:
    {selectedText}""",

    "custom": """TASK: Rewrite the following text according to the user's specific instructions provided below. Follow the instructions precisely and creatively to transform the text as requested.

    ORIGINAL TEXT:
    {selectedText}

    USER INSTRUCTIONS:
    {customPrompt}""",
}



rewrite_prompt = PromptTemplate( input_variables=["selectedText", "customPrompt"], template=rewrite_prompt_template ,
                                      partial_variables={"format_instructions": parser.get_format_instructions()})


chain = rewrite_prompt | llm | parser



def get_rewritten_text(selected_text, custom_prompt=""):
    """
    Generates rewritten text using the specified rewrite type.
    
    Args:
        rewrite_type (str): The type of rewriting (e.g., 'shorter', 'longer', 'more_intense', 'more_descriptive', 'custom').
        selected_text (str): The text that needs to be rewritten.
        custom_prompt (str): (Optional) Custom instructions for the 'custom' rewrite type.
        
    Returns:
        str: The rewritten text in JSON format.
    """
    # Choose the appropriate prompt template
    
    
    # Prepare the input for the chain
    input_data = {
        "selectedText": selected_text,
        "customPrompt": custom_prompt
    }
    
    # Generate the prompt using PromptTemplate
    
    
    # Invoke the chain and get the response
    response = chain.invoke(input_data)
    return response


