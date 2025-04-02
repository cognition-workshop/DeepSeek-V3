import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from model_wrapper import DeepSeekWrapper

app = FastAPI(title="Customer Support Ticket Analyzer")

templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "../"  # Root of the DeepSeek-V3 repository
CONFIG_PATH = "../inference/configs/config_671B.json"

model = None

def get_model():
    """Lazy initialization of the model to save resources"""
    global model
    if model is None:
        try:
            model = DeepSeekWrapper(MODEL_PATH, CONFIG_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return model

class TicketRequest(BaseModel):
    text: str

class TicketAnalysis(BaseModel):
    summary: str
    sentiment_score: int
    original_text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze", response_model=TicketAnalysis)
async def analyze_ticket(ticket: TicketRequest):
    """API endpoint to analyze a ticket"""
    if not ticket.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty")
    
    model = get_model()
    summary, sentiment_score = model.process_ticket(ticket.text)
    
    return TicketAnalysis(
        summary=summary,
        sentiment_score=sentiment_score,
        original_text=ticket.text
    )

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_form(request: Request, ticket_text: str = Form(...)):
    """Handle form submission from the web interface"""
    if not ticket_text.strip():
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": "Ticket text cannot be empty"}
        )
    
    model = get_model()
    summary, sentiment_score = model.process_ticket(ticket_text)
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": True,
            "summary": summary,
            "sentiment_score": sentiment_score,
            "original_text": ticket_text
        }
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
