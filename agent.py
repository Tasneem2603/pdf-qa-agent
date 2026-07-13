"""
Minimal PDF Q&A agent using LangGraph.

Flow: load PDF -> chunk it -> for each question, retrieve the most relevant
chunk(s) -> answer from them.

No LLM API key is available in this sandbox, so "answering" is done by
extractive retrieval (returning the most relevant sentence/passage) rather
than a generated free-text answer. This is honestly labelled below, and
swapping in a real model call is a one-line change — see ANSWER_WITH_LLM.
"""
import re
from typing import TypedDict, List, Optional
from pypdf import PdfReader
from langgraph.graph import StateGraph, END


def load_and_chunk_pdf(path: str) -> List[str]:
    """Extracts all text from the PDF and splits it into sentences. Sentence-level
    chunks give cleaner, more precise extractive answers than paragraph-level chunks.

    Some professionally-typeset PDFs (e.g. small-caps headings) encode certain
    glyphs in a way that pypdf extracts as literal null bytes / control characters,
    which silently corrupts nearby text if not stripped. Found this by testing on
    a real book PDF, not anticipated in advance — stripped here."""
    reader = PdfReader(path)
    full_text = " ".join(page.extract_text() for page in reader.pages)
    full_text = re.sub(r"[\x00-\x1f\x7f]", " ", full_text)  # replace control/null chars with a space
    full_text = re.sub(r"\s+", " ", full_text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def score_chunk(question: str, chunk: str) -> int:
    """Simple keyword-overlap relevance score: counts how many of the question's
    significant words (by 5-character prefix, so 'employees' still matches 'employs')
    appear in the chunk. Minimal on purpose — a real system would use embeddings +
    vector similarity (e.g. FAISS, Chroma) instead."""
    stopwords = {"the","a","an","is","are","was","were","what","who","when",
                 "where","how","does","did","do","in","on","of","to","and","for"}
    def prefixes(text):
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stopwords]
        return {w[:5] for w in words}
    return len(prefixes(question) & prefixes(chunk))


class AgentState(TypedDict):
    pdf_path: str
    question: str
    chunks: Optional[List[str]]
    retrieved: Optional[List[str]]
    answer: Optional[str]


def node_load(state: AgentState) -> AgentState:
    state["chunks"] = load_and_chunk_pdf(state["pdf_path"])
    return state


def node_retrieve(state: AgentState) -> AgentState:
    scored = [(score_chunk(state["question"], c), c) for c in state["chunks"]]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for score, c in scored[:3] if score > 0]
    state["retrieved"] = top if top else [scored[0][1]]
    return state


def node_answer(state: AgentState) -> AgentState:
    # ANSWER_WITH_LLM: to make this generative instead of extractive, replace this
    # function body with something like:
    #   prompt = f"Answer the question using only this context:\n{state['retrieved']}\n\nQuestion: {state['question']}"
    #   response = anthropic_client.messages.create(model="claude-sonnet-4-6", messages=[{"role":"user","content":prompt}])
    #   state["answer"] = response.content[0].text
    state["answer"] = " ".join(state["retrieved"])
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load", node_load)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.set_entry_point("load")
    g.add_edge("load", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


def build_graph_preloaded():
    """Same as build_graph, but skips the 'load' node — used when the PDF has
    already been read and chunked once, so repeated questions (e.g. in a chat
    loop) don't re-read and re-chunk the file every single time."""
    g = StateGraph(AgentState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


def ask(pdf_path: str, question: str) -> str:
    graph = build_graph()
    result = graph.invoke({"pdf_path": pdf_path, "question": question,
                            "chunks": None, "retrieved": None, "answer": None})
    return result["answer"]


class PDFAgent:
    """Loads a PDF once, then answers as many questions as you like against it
    without re-reading the file each time. This is what an interactive chat
    loop should use instead of calling ask() repeatedly."""
    def __init__(self, pdf_path: str):
        self.chunks = load_and_chunk_pdf(pdf_path)
        self.graph = build_graph_preloaded()

    def ask(self, question: str) -> str:
        result = self.graph.invoke({"pdf_path": None, "question": question,
                                     "chunks": self.chunks, "retrieved": None, "answer": None})
        return result["answer"]


if __name__ == "__main__":
    pdf_path = "demo.pdf"  # this is A Dog's Tale by Mark Twain, the actual assigned PDF
    questions = [
        "Who was the narrator's mother?",
        "What breed was the narrator's father?",
        "What was the name of the family that bought the dog?",
        "How did the dog save the baby?",
        "What happened to the puppy in the laboratory experiment?",
        "What was the puppy's name?",
        "What word does the mother dog use as her emergency word?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(pdf_path, q)}")
