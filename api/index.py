import os
from typing import TypedDict, Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

from langgraph.graph import StateGraph, END

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_NAME = "selfai1"
NAMESPACE = "__default__"
EMBED_MODEL = "models/gemini-embedding-001"  # dimension 3072
CHAT_MODEL = "gemini-3.5-flash-lite"

_pinecone_index = None
_gemini_model = None
_embed_configured = False

def get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index

def _ensure_genai():
    global _embed_configured
    if not _embed_configured:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _embed_configured = True

def get_model():
    global _gemini_model
    _ensure_genai()
    if _gemini_model is None:
        _gemini_model = genai.GenerativeModel(CHAT_MODEL)
    return _gemini_model

class GraphState(TypedDict, total=False):
    question: str
    history: List[Dict[str, str]]
    embedding: Optional[List[float]]
    context: str
    answer: str
    error: Optional[str]

def embed_node(state: GraphState) -> GraphState:
    """Node 1: ubah pertanyaan user menjadi vector pakai Gemini Embedding."""
    if state.get("error"):
        return state
    try:
        _ensure_genai()
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=state["question"],
            task_type="retrieval_query",
        )
        state["embedding"] = result["embedding"]
    except Exception as e:
        state["error"] = f"[Gemini Embedding] {e}"
    return state

def retrieve_node(state: GraphState) -> GraphState:
    """Node 2: query Pinecone pakai vector hasil embed."""
    if state.get("error"):
        return state
    try:
        index = get_index()
        results = index.query(
            namespace=NAMESPACE,
            vector=state["embedding"],
            top_k=10,
            include_metadata=True,
        )
        contexts = [
            match["metadata"].get("text", "")
            for match in results["matches"]
            if match.get("metadata") and match["metadata"].get("text")
        ]
        state["context"] = "\n\n".join(contexts) if contexts else "Tidak ada dokumen relevan."
    except Exception as e:
        state["error"] = f"[Pinecone] {e}"
    return state

def generate_node(state: GraphState) -> GraphState:
    """Node 3: generate jawaban pakai Gemini, dengan rules persis seperti versi lama."""
    if state.get("error"):
        return state
    try:
        model = get_model()
        rules = (
            "You are Bintang Eko Pratomo, an AI assistant speaking in first person ('I') to a recruiter or visitor. "
            "Never refer to yourself in third person.\n"
            "Use the context below as source facts only — paraphrase naturally, never copy verbatim. "
            "Preserve exact facts (names, numbers, tools, dates); never add info not in the context. "
            "Ignore metadata, URLs, IDs, and formatting symbols in the context.\n"
            "If it's just a greeting or small talk (hi, how are you), reply casually without using the context. "
            "If the user shares something personal or off-topic (e.g. feeling sick, venting, casual chit-chat unrelated to work), "
            "respond briefly and empathetically, then gently steer back by mentioning this is a portfolio assistant and inviting "
            "them to ask about experience, skills, or projects.\n"
            "If it's about skills, experience, projects, or contact, answer using the context. "
            "If it's a factual question and the answer isn't in the context, say the info isn't available — don't guess.\n"
            "Answer concisely, in the same language as the question, in a natural conversational tone — not like reading a CV.\n\n"
            f"Context:\n{state.get('context', '')}\n\n"
            f"Question: {state['question']}\n"
            "Answer:"
        )

        recent_history = state.get("history", [])[-10:]
        gemini_contents = [
            {"role": h["role"], "parts": [h["content"]]} for h in recent_history
        ]
        gemini_contents.append(
            {"role": "user", "parts": [rules]}
        )

        response = model.generate_content(gemini_contents)
        state["answer"] = response.text
    except Exception as e:
        state["error"] = f"[Gemini Generation] {e}"
    return state

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("embed", embed_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("embed")
    workflow.add_edge("embed", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

rag_graph = build_graph()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

@app.get("/")
def home():
    return {"status": "API Chatbot Active! (LangGraph)"}

@app.post("/api/chat")
def chat_bot(req: ChatRequest):
    try:
        try:
            get_index()
        except Exception as e:
            return {"error": f"[Pinecone Init] {e}"}
        try:
            get_model()
        except Exception as e:
            return {"error": f"[Gemini Init] {e}"}

        initial_state: GraphState = {
            "question": req.question,
            "history": [h.dict() for h in req.history],
        }
        final_state = rag_graph.invoke(initial_state)

        if final_state.get("error"):
            return {"error": final_state["error"]}

        return {"response": final_state.get("answer", "")}

    except Exception as e:
        return {"error": str(e)}
