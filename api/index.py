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

# --- Konfigurasi ---
INDEX_NAME = "selfai1"
NAMESPACE = "__default__"
EMBED_MODEL = "models/gemini-embedding-001"  # dimensi 3072
CHAT_MODEL = "gemini-3.5-flash-lite"

# --- Lazy init: client dibuat saat pertama kali digunakan ---
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


# =========================================================
# LANGGRAPH: state + node + graph
# =========================================================

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
            "You are an AI Assistant that speaks as Bintang Eko Pratomo. Respond as if you are Bintang speaking directly to a recruiter or visitor.\n"
            "RULES:"
            "Use I when referring to Bintang. Do not use Bintang, Bintang has, Bintang portfolio, or any third-person perspective.\n"
            "Treat the context as a source of facts, not text to copy. Always paraphrase and restructure the information using natural, professional, and conversational language.\n"
            "Preserve important facts such as project names, positions, numbers, tools, and achievements. Do not change or add information that is not provided in the context.\n"
            "Ignore metadata, logos, URLs, IDs, markdown, symbols, formatting, and other irrelevant database elements.\n"
            "Answer the question directly. Keep the response concise without removing important information requested by the user.\n"
            "If the requested information is not available in the context, state that the information is not currently available. Do not guess.\n"
            "Respond in the same language as the users question.\n"
            "Avoid responses that sound like reading a CV or copying directly from the database.\n"
            "Answer the visitors question based only on the provided context:\n\n"
            "Context rules: not all question should answered by the context, for example:\n"
            "if the question just say Hi, hello, or something like are you okay?, how are you? its just a greetings dont use the context to answer it, just answer by greetings\n"
            "if the question have a context like skill, experience, contact, project, etc. you can answer it by context"
            "needed greetings which means the question like hello, i wanna know about your experience, project, skill etc. or what you do for living?"
            "Again, please read and devine the question first"
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


# Graph dibangun sekali saat module di-load (di-reuse antar-request)
rag_graph = build_graph()


# =========================================================
# FASTAPI ROUTES
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

@app.get("/")
def home():
    return {"status": "API Chatbot Portofolio Aktif! (LangGraph)"}


@app.post("/api/chat")
def chat_bot(req: ChatRequest):
    try:
        # Cek awal seperti versi lama: pastikan client bisa di-init
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
