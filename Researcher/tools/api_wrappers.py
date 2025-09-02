from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from config import DOC_CONTENT_MAX_CHARS


def wiki_tool(topic: str) -> str:
    """Search Wikipedia for the given topic and return the most relevant page content.

    Args:
        topic (str): The subject or keyword to search on Wikipedia.

    Returns:
        str: Top three retrieved Wikipedia articles, including title, source, and content. Separated by `---`.
    """
    try:
        wiki = WikipediaAPIWrapper(
            top_k_results= 3,
            lang= 'en',
            doc_content_chars_max= DOC_CONTENT_MAX_CHARS
        )
        docs = wiki.load(topic)

        output = []
        for index, doc in enumerate(docs, start = 1):
            output.append(
                f"Index: {index}\n"
                f"Title: {doc.metadata.get('title', 'Unknown')}\n"
                f"Source: {doc.metadata.get('source', 'Wikipedia')}\n"
                f"Content: {doc.page_content.strip()}\n"
                "\n---\n"
            )

        return "\n".join(output) if output else "No Wikipedia articles found."
        
    except Exception as e:
        return f"error: Wikipedia search failed: {str(e)}"


def arxiv_tool(topic: str) -> str:
    """Search Arxiv for academic papers related to the given topic.

    Args:
        topic (str): The research subject or keyword to query on Arxiv.

    Returns:
        str: Top three retrieved research papers, including title, publishing date, authors, source and content. Separated by `---`.
    """
    try:
        arxiv = ArxivAPIWrapper(
            top_k_results= 3,
            doc_content_chars_max= DOC_CONTENT_MAX_CHARS
        )
        docs = arxiv.load(topic)

        output = []
        for index, doc in enumerate(docs, start = 1):
            output.append(
                f"Index: {index}\n"
                f"Title: {doc.metadata.get('Title', 'Unknown')}\n"
                f"Published: {doc.metadata.get('Published')}\n"
                f"Authors: {doc.metadata.get('Authors')}\n"
                f"Source: {doc.metadata.get('Source', 'Arxiv research paper')}\n"
                f"Content: {doc.page_content.strip()}\n"
                "\n---\n"
            )

        return "\n".join(output) if output else "No Arxiv papers found."
        
    except Exception as e:
        return f"error: Arixv search failed: {str(e)}"
