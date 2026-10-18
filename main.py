import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

app = FastAPI(title="Dental Transcriptor")
MAX_AUDIO_BYTES = 25 * 1024 * 1024

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as page:
        return page.read()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="Falta configurar la clave del servicio.")
    mime = (audio.content_type or "").split(";")[0].lower().strip()
    allowed = {"audio/webm", "audio/mp4", "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/ogg", "audio/aac", "audio/flac", "audio/m4a"}
    if mime not in allowed:
        raise HTTPException(status_code=415, detail="Formato de audio no compatible.")
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    await audio.close()
    if not data:
        raise HTTPException(status_code=400, detail="La grabación está vacía.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="El audio supera el límite de 25 MB.")
    prompt = """Transcribe literalmente y en español el audio dictado por un odontólogo después de una consulta. Devuelve únicamente la transcripción, sin títulos, resúmenes, interpretaciones ni formato de historia clínica. Conserva el contenido, orden y sentido exactos. Reconoce términos odontológicos, procedimientos, materiales, medicamentos y anatomía dental. Conserva los números de dientes tal como se dictan. No inventes ni completes datos por contexto. Si algo no se entiende, escribe [inaudible]. Usa puntuación razonable sin reescribir."""
    try:
        client = genai.Client(api_key=key)
        result = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=[prompt, types.Part.from_bytes(data=data, mime_type=mime)],
            config=types.GenerateContentConfig(temperature=0),
        )
        text = (result.text or "").strip()
        if not text:
            raise HTTPException(status_code=502, detail="El servicio no devolvió una transcripción.")
        return {"transcript": text}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="No fue posible transcribir el audio. Inténtalo de nuevo.")
    finally:
        del data
