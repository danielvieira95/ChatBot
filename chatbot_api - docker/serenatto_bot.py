import os
from typing import List
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.simple import SimpleVectorStore  # ✅ correto
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatSummaryMemoryBuffer


class SerenattoBot:
    def __init__(self):
        self.model_name = "intfloat/multilingual-e5-large"

        # Tokenizer e modelo HuggingFace com truncamento seguro
        tokenizer = AutoTokenizer.from_pretrained(
        self.model_name,
        truncation=True,
        padding=True,
        max_length=512,
        force_download=True  # <- aqui está certo
    )

        model = AutoModel.from_pretrained(
        self.model_name,
        force_download=True  # <- aqui está certo também
    )

        self.embed_model = HuggingFaceEmbedding(
        model=model,
        tokenizer=tokenizer,
        embed_batch_size=16  # <- aqui sem 'force_download'
    )
        self.llm = Groq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
        )

        self.chat_engine = None
        self._inicializar_index()

    def _extrair_texto_pdf(self, caminho_pdf: str) -> str:
        if not os.path.exists(caminho_pdf):
            raise FileNotFoundError(f"PDF não encontrado: {caminho_pdf}")
        reader = PdfReader(caminho_pdf)
        return "\n".join((page.extract_text() or "").strip() for page in reader.pages)

    def _inicializar_index(self):
        texto = self._extrair_texto_pdf("documentos/serenatto.pdf")

        with open("documentos/temp.txt", "w", encoding="utf-8") as f:
            f.write(texto)

        documentos = SimpleDirectoryReader(input_dir="documentos")
        docs = documentos.load_data()

        node_parser = SentenceSplitter(chunk_size=800)
        nodes = node_parser.get_nodes_from_documents(docs)

        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model
        )

        memory = ChatSummaryMemoryBuffer(llm=self.llm, token_limit=256)

        self.chat_engine = index.as_chat_engine(
            chat_mode="context",
            llm=self.llm,
            memory=memory,
            system_prompt="Você é especialista em cafés da loja Serenatto. Responda de forma simpática e natural sobre os grãos disponíveis."
        )

    def responder(self, mensagem: str) -> str:
        if not self.chat_engine:
            return "Bot ainda não está pronto."
        resposta = self.chat_engine.chat(mensagem)
        return resposta.response

    def resetar(self):
        if self.chat_engine:
            self.chat_engine.reset()
