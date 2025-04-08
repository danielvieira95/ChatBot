from fastapi import FastAPI, Body
from pydantic import BaseModel
from serenatto_bot import SerenattoBot

app = FastAPI(title="Serenatto Chatbot API")

# Instancia o bot na inicialização
bot = SerenattoBot()

class MensagemRequest(BaseModel):
    mensagem: str

class MensagemResponse(BaseModel):
    resposta: str

@app.post("/conversar", response_model=MensagemResponse)
def conversar(request: MensagemRequest):
    resposta = bot.responder(request.mensagem)
    return MensagemResponse(resposta=resposta)

@app.post("/resetar")
def resetar():
    bot.resetar()
    return {"status": "Chat resetado com sucesso"}
