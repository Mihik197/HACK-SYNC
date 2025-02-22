from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tts import play_story_from_json
from imageGen import generate_and_save_image
from brainStorming import get_brainstorming_ideas
import uvicorn

app = FastAPI()

class StoryRequest(BaseModel):
    story: str

class ImageRequest(BaseModel):
    prompt: str

class IdeaRequest(BaseModel):
    category: str
    list_of: str
    context: str = None
    examples: str = None

@app.post("/tts/")
async def text_to_speech(request: StoryRequest):
    try:
        json_input = request.json()
        play_story_from_json(json_input)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/imageGen/")
async def generate_image(request: ImageRequest):
    try:
        json_input = request.json()
        result = generate_and_save_image(json_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/brainstorming/")
async def brainstorming(request: IdeaRequest):
    try:
        json_input = request.dict()
        result = get_ideas(json_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)