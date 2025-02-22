from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

# Import your modules
from tts import play_story_from_json
from imageGen import generate_and_save_image
from brainStorming import get_brainstorming_ideas
from chapter import generate_story_chapter
from character import generate_character_profile
from outline import generate_plot_outline
from quick_edit import perform_quick_edit
from rewrite import get_rewritten_text

import uvicorn
import json  # Necessary for imageGen and tts

app = FastAPI(
    title="AI Storytelling API",
    description="A collection of APIs for generating and manipulating story content.",
    version="1.0.0",
)


class StoryRequest(BaseModel):
    story: str = Field(..., description="The story text to be converted to speech.")


class ImageRequest(BaseModel):
    prompt: str = Field(..., description="The prompt for image generation.")


class IdeaRequest(BaseModel):
    category: str = Field(..., description="Category for brainstorming.")
    list_of: str = Field(..., description="The type of ideas to generate.")
    context: Optional[str] = Field(None, description="Optional context for brainstorming.")
    examples: Optional[str] = Field(None, description="Optional examples for brainstorming.")


class ChapterRequest(BaseModel):
    plot_point: str = Field(..., description="The main event or development for this chapter.")
    previous_chapters: str = Field(..., description="Text of the previous chapters for context.")
    character_data: str = Field(..., description="Description of characters involved in the story.")
    worldbuilding_data: str = Field(..., description="Information about the story's world, setting, and rules.")
    user_genre: str = Field(..., description="The genre of the story (e.g., Fantasy, Romance, Thriller).")
    user_style: str = Field(..., description="User-specified style, including length or stylistic preferences.")


class CharacterRequest(BaseModel):
    user_character_description: str = Field(..., description="Description of the character's role, appearance, or other traits.")
    user_genre: str = Field(..., description="The genre of the story (e.g., Fantasy, Romance, Thriller).")


class OutlineRequest(BaseModel):
    user_premise: str = Field(..., description="The main idea or scenario of the story.")
    user_genre: str = Field(..., description="The genre of the story (e.g., Fantasy, Romance, Thriller).")


class QuickEditRequest(BaseModel):
    user_request: str = Field(..., description="Specific editing request from the user.")
    document_text: str = Field(..., description="The current story text to be edited.")
    character_data: str = Field(..., description="Character information for context.")
    worldbuilding_data: str = Field(..., description="Worldbuilding details for consistency.")
    user_genre: str = Field(..., description="The genre of the story (e.g., Fantasy, Romance, Thriller).")


class RewriteRequest(BaseModel):
    selected_text: str = Field(..., description="The text to be rewritten.")
    rewrite_type: str = Field(..., description="The type of rewriting (shorter, longer, more_intense, more_descriptive, custom).")
    custom_prompt: Optional[str] = Field(None, description="Optional custom instructions for rewriting.")

# TTS Endpoint
@app.post("/tts/", tags=["Text-to-Speech"])
async def text_to_speech(request: StoryRequest):
    """Converts story text to speech."""
    try:
        json_input = request.json()  # Access the raw JSON string
        play_story_from_json(json_input)
        return {"status": "success", "message": "Story is now playing."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Image Generation Endpoint
@app.post("/imageGen/", tags=["Image Generation"])
async def generate_image(request: ImageRequest):
    """Generates an image from a text prompt."""
    try:
        json_input = request.json()  # Access the raw JSON string
        result = generate_and_save_image(json_input)
        return json.loads(result)  # Return result as a dictionary
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Brainstorming Endpoint
@app.post("/brainstorming/", tags=["Brainstorming"])
async def brainstorming(request: IdeaRequest):
    """Generates brainstorming ideas."""
    try:
        result = get_brainstorming_ideas(
            category=request.category,
            list_of=request.list_of,
            context=request.context,
            examples=request.examples,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Chapter Generation Endpoint
@app.post("/chapter/", tags=["Chapter Generation"])
async def chapter_generation(request: ChapterRequest):
    """Generates a story chapter."""
    try:
        chapter = generate_story_chapter(
            plot_point=request.plot_point,
            previous_chapters=request.previous_chapters,
            character_data=request.character_data,
            worldbuilding_data=request.worldbuilding_data,
            user_genre=request.user_genre,
            user_style=request.user_style,
        )
        return chapter
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Character Generation Endpoint
@app.post("/character/", tags=["Character Generation"])
async def character_generation(request: CharacterRequest):
    """Generates a character profile."""
    try:
        profile = generate_character_profile(
            user_character_description=request.user_character_description,
            user_genre=request.user_genre,
        )
        return profile
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Outline Generation Endpoint
@app.post("/outline/", tags=["Outline Generation"])
async def outline_generation(request: OutlineRequest):
    """Generates a plot outline."""
    try:
        outline = generate_plot_outline(
            user_premise=request.user_premise,
            user_genre=request.user_genre,
        )
        return outline
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Quick Edit Endpoint
@app.post("/quick_edit/", tags=["Quick Edit"])
async def quick_edit(request: QuickEditRequest):
    """Performs a quick edit on story text."""
    try:
        edited_text = perform_quick_edit(
            user_request=request.user_request,
            document_text=request.document_text,
            character_data=request.character_data,
            worldbuilding_data=request.worldbuilding_data,
            user_genre=request.user_genre,
        )
        return edited_text
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Rewrite Endpoint
# Rewrite Endpoint
@app.post("/rewrite/", tags=["Rewrite"])
async def rewrite(request: RewriteRequest):
    """Rewrites text based on instructions."""
    try:
        rewritten_text = get_rewritten_text(
            selected_text=request.selected_text,
            rewrite_type=request.rewrite_type,
            custom_prompt=request.custom_prompt,
        )
        return rewritten_text
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)