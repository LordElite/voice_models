from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from voice_models import text_to_cloned_speech

# ---------------------
# FASTAPI APP
# ---------------------
app = FastAPI()

@app.post("/tts")
async def tts_endpoint(text: str = Form(...)):
    """
    Convierte texto en audio con la voz clonada.
    """
    save_path = text_to_cloned_speech(text, "output.wav")
    return FileResponse(save_path, media_type="audio/wav", filename="output.wav")

