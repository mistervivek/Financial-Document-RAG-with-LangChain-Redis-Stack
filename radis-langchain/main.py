"""
Financial Document RAG with LangChain, Redis Stack, and OpenRouter.

Author: Vivek
Repository: redis-langchain
"""

import argparse
import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress noisy warnings
warnings.filterwarnings("ignore")
load_dotenv(override=True)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from redisvl.index import SearchIndex


INDEX_NAME = "financial_10k_rag"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_MODEL", "apodex/apodex-1.1-mini:free")


def get_redis_url() -> str:
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")
    if password:
        return f"redis://:{password}@{host}:{port}"
    return f"redis://{host}:{port}"


def get_llm(model: str = DEFAULT_LLM_MODEL) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        print("[ERROR] Valid OPENROUTER_API_KEY is not set. Please set your key in .env or your environment.")
        sys.exit(1)

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1", # Enforces OpenRouter routing over OpenAI
        default_headers={
            "HTTP-Referer": "https://github.com/radis-langchain",
            "X-Title": "Financial Document RAG",
        },
        temperature=0.2,
    )


def load_and_index_documents(
    doc_path: str,
    redis_url: str = None,
    reindex: bool = False,
) -> RedisVectorStore:
    if redis_url is None:
        redis_url = get_redis_url()

    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)

    if not reindex:
        try:
            # Check if index exists
            existing_idx = SearchIndex.from_existing(INDEX_NAME, redis_url=redis_url)
            if existing_idx.exists():
                print(f"[INFO] Using existing Redis vector index: '{INDEX_NAME}'")
                return RedisVectorStore(
                    embeddings=embeddings,
                    index_name=INDEX_NAME,
                    redis_url=redis_url,
                )
        except Exception:
            pass

    print(f"[INFO] Indexing document from: {doc_path} ...")
    if not os.path.exists(doc_path):
        print(f"[ERROR] Document file not found at: {doc_path}")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
    loader = UnstructuredLoader(
        doc_path,
        chunking_strategy="basic",
        max_characters=1000000,
        include_orig_elements=False,
        strategy="fast",
    )
    chunks = loader.load_and_split(text_splitter)
    print(f"[INFO] Extracted {len(chunks)} text chunks.")

    print(f"[INFO] Storing embeddings into Redis Stack at {redis_url} ...")
    vector_store = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        redis_url=redis_url,
        metadata_schema=[{"name": "source", "type": "text"}],
    )
    print(f"[SUCCESS] Vector store ready! Total records: {vector_store._index.client.dbsize()}")
    return vector_store


def build_rag_chain(vector_store: RedisVectorStore, llm: ChatOpenAI):
    prompt_template = """Use the following pieces of context from financial 10-K filings to answer the question at the end.
If you do not know the answer based on the context, say that you don't know; do not try to make up an answer.

Context:
---------
{context}
---------

Question: {question}
Answer:"""

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {
            "context": vector_store.as_retriever(search_kwargs={"k": 4}) | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main():
    parser = argparse.ArgumentParser(description="Financial Document RAG with LangChain & Redis")
    parser.add_argument("--query", "-q", type=str, help="Question to ask the RAG system")
    parser.add_argument("--doc", "-d", type=str, default="resources/nke-10k-2023.pdf", help="Path to PDF filing")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of documents in Redis")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive Q&A mode")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_LLM_MODEL, help="OpenRouter model name")
    args = parser.parse_args()

    redis_url = get_redis_url()
    print("=" * 60)
    print(" Financial Document RAG Pipeline (Redis + LangChain)")
    print("=" * 60)
    print(f" Redis Endpoint:  {redis_url}")
    print(f" LLM Model:       {args.model}")
    print("=" * 60)

    # Initialize components
    vector_store = load_and_index_documents(doc_path=args.doc, redis_url=redis_url, reindex=args.reindex)
    llm = get_llm(model=args.model)
    rag_chain = build_rag_chain(vector_store, llm)

    if args.interactive:
        print("\n[INFO] Entering interactive mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                question = input("Ask a question: ").strip()
                if not question:
                    continue
                if question.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                print("\nThinking...")
                answer = rag_chain.invoke(question)
                print(f"\nAnswer:\n{answer}\n")
                print("-" * 60)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.query:
        print(f"\nQuestion: {args.query}")
        print("\nThinking...")
        answer = rag_chain.invoke(args.query)
        print(f"\nAnswer:\n{answer}\n")
    else:
        sample_query = "What was Nike's revenue last year compared to this year?"
        print(f"\n[INFO] No query specified. Running sample query: '{sample_query}'\n")
        answer = rag_chain.invoke(sample_query)
        print(f"Answer:\n{answer}\n")


if __name__ == "__main__":
    main()