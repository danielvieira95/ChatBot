# Inclusão das bibliotecas
import os
from typing import List
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.simple import SimpleVectorStore  # substituto leve e sem compilação
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatSummaryMemoryBuffer
from tempfile import TemporaryDirectory
from PyPDF2 import PdfReader
# pip install llama-index==0.10.30


class SerenattoBot:
    def __init__(self):
        # Modelo de embedding
        self.embed_model = HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')

        # Vetor store simples (100% Python, sem dependência de C++)
        self.vector_store = SimpleVectorStore()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # LLM via Groq
        self.llms = Groq(
            model='llama3-70b-8192',
            api_key='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'
        )

        self.document_index = None
        self.chat_engine = None

        self.carregar_pdf()

    def carregar_pdf(self):
        with TemporaryDirectory() as tmpdir:
            pdf_path = "documentos/serenatto.pdf"
            text = ""
            reader = PdfReader(pdf_path)

            for page in reader.pages:
                text += page.extract_text() or ""

            with open(os.path.join(tmpdir, "temp.txt"), "w", encoding="utf-8") as f:
                f.write(text)

            documentos = SimpleDirectoryReader(input_dir=tmpdir)
            docs = documentos.load_data()

            node_parser = SentenceSplitter(chunk_size=1200)
            nodes = node_parser.get_nodes_from_documents(docs)

            self.document_index = VectorStoreIndex(
                nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )

            memory = ChatSummaryMemoryBuffer(llm=self.llms, token_limit=256)

            self.chat_engine = self.document_index.as_chat_engine(
                chat_mode='context',
                llm=self.llms,
                memory=memory,
                system_prompt='''Você é especialista em cafés da loja Serenatto. 
                Responda de forma simpática e natural sobre os grãos disponíveis.'''
            )

    def responder(self, mensagem: str) -> str:
        if self.chat_engine is None:
            return "Erro: o bot ainda não está pronto."
        response = self.chat_engine.chat(mensagem)
        return response.response

    def resetar(self):
        if self.chat_engine:
            self.chat_engine.reset()
