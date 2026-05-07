from typing import Iterator, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

_SYSTEM_PROMPT = """You are a precise document assistant. Answer the question using only \
the context provided below. If the context does not contain enough information, say so \
clearly rather than guessing. Always mention which document(s) your answer comes from.

Context:
{context}"""


def _format_docs(docs: List[Document]) -> str:
    sections = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        header = f"[{i}] {filename}" + (f", page {page + 1}" if page is not None else "")
        sections.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(sections)


def build_rag_chain(retriever, model: str, base_url: str, temperature: float = 0.1):
    """
    Builds a retrieval-augmented generation chain using LangChain's expression
    language (LCEL). The retriever fetches relevant chunks, formats them into
    the prompt context, and Ollama generates the grounded answer.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_answer(chain, question: str) -> str:
    return chain.invoke(question)


def stream_answer(chain, question: str) -> Iterator[str]:
    return chain.stream(question)
