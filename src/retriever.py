import os
import re
import glob
import shutil

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads NVIDIA_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    # give every chunk an id like "s3-c12" (session 3, chunk 12) for the eval later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"s{chunk.metadata['session']}-c{i}"

    store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # embed in small batches so we stay under the free-tier rate limit
    try:
        for i in range(0, len(chunks), 32):
            store.add_documents(chunks[i : i + 32])
    except Exception:
        shutil.rmtree(DB_DIR, ignore_errors=True)  # don't keep a half-built store
        raise

    return store


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python -m src.retriever
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")