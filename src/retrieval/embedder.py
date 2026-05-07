from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load a sentence-transformers embedding model that runs fully locally.
    The model is downloaded once and cached by HuggingFace hub.

    all-MiniLM-L6-v2 is a good default — small (80 MB), fast, and
    produces decent semantic embeddings for English text.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
