from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

llm = ChatOpenAI(
    base_url='https://api.together.xyz/v1',
    api_key=TOGETHER_API_KEY,
    model='google/gemma-2-27b-it',
    temperature=1
)

class IDEAS(BaseModel):
    idea: str = Field(description="ideas for the given context")

parser = JsonOutputParser(pydantic_object=IDEAS)

from langchain_core.prompts import PromptTemplate

brainstorming_prompt_template = """
You are a creative brainstorming assistant, skilled at generating a wide range of ideas, concepts, and solutions. Your goal is to help users explore possibilities by offering diverse and imaginative suggestions. Follow these steps:

1. **Identify the Category:** 
   - The user selects one of the following categories: 
     - Characters, World Building, Plot Points, Names, Places, Objects, Descriptions, Article Ideas, Tweets.
   - This choice guides the type of ideas to be generated, but it should **not** be included in the output.

2. **Understand the Request:** 
   - The user specifies what they want a list of (e.g., character backstories, plot twists, world-building elements).
   - If context or examples are provided, use them to guide the brainstorming.
   - If not, proceed with general ideation relevant to the category.

3. **Generate Ideas:** 
   - Create a diverse list of ideas relevant to the chosen category and request.
   - Ensure the ideas are imaginative, varied, and thought-provoking.

4. **Organize and Present:** 
   - Present the ideas in a clear, organized list.
   - Keep each idea concise yet vivid.
   - The response should be in JSON format.
   - The output **must not** include the category.

**Parameters:**

*   **Category:**  {category}  (Choose one: Characters, World Building, Plot Points, Names, Places, Objects, Descriptions, Article Ideas, Tweets)
*   **Give me a list of:**  {list_of} (e.g., "Backstory ideas for a character who is emotionally guarded.")
*   **Context (optional):**  {context} (e.g., "The character struggles with trusting others due to past experiences.")
*   **Examples (optional):** {examples} (e.g., "Julie was bullied in school for being different and the experience made her toughen up to protect herself.\\nSomeone she considered a close friend betrayed her trust, causing Julie to become more guarded.")

**Output Format:**
The ideas should be presented as a JSON object with the key "ideas", structured as follows:

```json
{{
    "ideas": [
        "Idea 1",
        "Idea 2",
        "Idea 3",
        "...and so on"
    ]
}}
"""

brainstorming_prompt = PromptTemplate( input_variables=["category", "list_of", "context", "examples"], template=brainstorming_prompt_template ,
                                      partial_variables={"format_instructions": parser.get_format_instructions()})



chain = brainstorming_prompt | llm | parser

def get_brainstorming_ideas(category, list_of, context="", examples=""):
    test_input = {
        "category": category,
        "list_of": list_of,
        "context": context,
        "examples": examples
    }
    response = chain.invoke(test_input)
    return response