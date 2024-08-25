from langgraph.graph import StateGraph, MessagesState, END
from typing import TypedDict
import requests
from xml.etree import ElementTree
from urllib.parse import quote


class ConxualizedKeyword(TypedDict):
    keyword: str
    local_context: str
    # global_context: str


class ConxualizedCitation(TypedDict):
    citation: str
    local_context: str
    # global_context: str


class ContextualizedCitationsAbstracts(TypedDict):
    citations: str
    context: str
    abstract: str


class ConxualizedKeywordList(TypedDict):
    keywords: list[ConxualizedKeyword]


class ConxualizedCitationList(TypedDict):
    citations: list[ConxualizedCitation]


# Define the state with a built-in messages key
class ResearchState(MessagesState):
    user_intent: str
    paper_url: str
    paper_md: str
    keywords: ConxualizedKeywordList
    citations: ConxualizedCitationList
    reading_assistance_md: str


def get_arxiv_paper_details(title, authors=None, year=None):
    base_url = "http://export.arxiv.org/api/query"

    # Construct the query
    query_parts = [f'ti:"{quote(title)}"']
    if authors:
        author_query = " AND ".join(f'au:"{quote(author)}"' for author in authors)
        query_parts.append(f"({author_query})")
    if year:
        query_parts.append(f"submittedDate:[{year}0101 TO {year}1231]")

    query = " AND ".join(query_parts)

    response = requests.get(f"{base_url}?search_query={query}")

    if response.status_code == 200:
        root = ElementTree.fromstring(response.content)
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            paper_title = entry.find("{http://www.w3.org/2005/Atom}title").text
            authors = [
                author.find("{http://www.w3.org/2005/Atom}name").text
                for author in entry.findall("{http://www.w3.org/2005/Atom}author")
            ]
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
            published_date = entry.find("{http://www.w3.org/2005/Atom}published").text
            year = published_date.split("-")[0]
            return {
                "title": paper_title,
                "authors": authors,
                "abstract": abstract,
                "year": year,
            }
    else:
        return None


# Define the logic for each node
def input_node(state: ResearchState) -> ResearchState:
    # Logic to process the input paper
    paper = "markdown"

    return {"paper_md": paper}


def keyword_extraction_node(state: ResearchState) -> ResearchState:
    # Logic to extract keywords
    response = llm.structuredOutput(ConxualizedKeywordList)
    return {"keywords": response}


def citation_extraction_node(state: ResearchState) -> ResearchState:
    # Logic to extract citations
    return {"citations": ["citation1", "citation2"]}


def contextualization_node(state: ResearchState) -> ResearchState:
    # Logic to add context to keywords and citations
    return {"context": "contextualized information"}


def abstract_fetching_node(state: ResearchState) -> ResearchState:
    # Logic to fetch abstracts for citations
    return {"abstracts": ["abstract1", "abstract2"]}


def reading_assistance_node(state: ResearchState) -> ResearchState:
    # Logic to provide reading assistance
    return {"reading_assistance": "assistance context"}


def final_contextualization_node(state: ResearchState) -> ResearchState:
    # Logic to finalize the contextualization
    return {"messages": [("system", "Final contextualization complete")]}


# Create the graph
graph = StateGraph(ResearchState)

# Add nodes to the graph
graph.add_node("input_node", input_node)
graph.add_node("keyword_extraction_node", keyword_extraction_node)
graph.add_node("citation_extraction_node", citation_extraction_node)
graph.add_node("contextualization_node", contextualization_node)
graph.add_node("abstract_fetching_node", abstract_fetching_node)
graph.add_node("reading_assistance_node", reading_assistance_node)
graph.add_node("final_contextualization_node", final_contextualization_node)

# Define the edges between nodes
graph.set_entry_point("input_node")
graph.add_edge("input_node", "keyword_extraction_node")
graph.add_edge("input_node", "citation_extraction_node")
graph.add_edge("keyword_extraction_node", "contextualization_node")
graph.add_edge("citation_extraction_node", "contextualization_node")
graph.add_edge("citation_extraction_node", "abstract_fetching_node")
graph.add_edge("contextualization_node", "reading_assistance_node")
graph.add_edge("abstract_fetching_node", "reading_assistance_node")
graph.add_edge("reading_assistance_node", "final_contextualization_node")
graph.set_finish_point("final_contextualization_node")

# Compile the graph
app = graph.compile()


url = "https://arxiv.org/pdf/2310.04406"
response = app.invoke({"paper_url": url})
print(response)