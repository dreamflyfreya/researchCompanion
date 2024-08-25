from langgraph.graph import StateGraph, MessagesState
from typing import TypedDict, Annotated
import requests
from xml.etree import ElementTree
from urllib.parse import quote
import aiohttp
import asyncio
from operator import add
from langgraph.constants import Send


class ConxualizedKeyword(TypedDict):
    keyword: str
    local_context: str
    # global_context: str


class ConxualizedCitation(TypedDict):
    citation: str
    local_context: str
    # global_context: str


class ContextualizedCitationsAbstract(TypedDict):
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
    abstracts: Annotated[list[ContextualizedCitationsAbstract], add]
    reading_assistance_md: str


async def get_arxiv_paper_details(title):
    base_url = "http://export.arxiv.org/api/query"

    # Construct the query using only the title
    query = f'ti:"{quote(title)}"'

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}?search_query={query}") as response:
            if response.status == 200:
                content = await response.text()
                root = ElementTree.fromstring(content)
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                    paper_title = entry.find("{http://www.w3.org/2005/Atom}title").text
                    paper_authors = [
                        author.find("{http://www.w3.org/2005/Atom}name").text
                        for author in entry.findall(
                            "{http://www.w3.org/2005/Atom}author"
                        )
                    ]
                    abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
                    published_date = entry.find(
                        "{http://www.w3.org/2005/Atom}published"
                    ).text
                    paper_year = published_date.split("-")[0]
                    return {
                        "title": paper_title,
                        "authors": paper_authors,
                        "abstract": abstract,
                        "year": paper_year,
                    }
            return None


async def fetch_abstract(
    citation: ConxualizedCitation,
) -> ContextualizedCitationsAbstract:
    # Simulate fetching abstract (replace with actual API call)
    await asyncio.sleep(1)  # Simulating network delay
    return {
        "citation": citation["citation"],
        "context": citation["local_context"],
        "abstract": f"Abstract for {citation['citation']}",
    }


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


async def abstract_fetching_node(state: ResearchState) -> dict:
    citations = state["citations"]["citations"]

    # Use asyncio.gather to fetch abstracts concurrently
    abstracts = await asyncio.gather(
        *[fetch_abstract(citation) for citation in citations]
    )

    # Return the list of abstracts to be combined using the `add` reducer
    return {"abstracts": abstracts}


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
