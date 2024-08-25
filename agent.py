from langgraph.graph import StateGraph, MessagesState
from typing import TypedDict, Annotated
import requests
from xml.etree import ElementTree
from urllib.parse import quote
import aiohttp
import asyncio
from operator import add
from langgraph.constants import Send
from langchain_groq import ChatGroq
import pymupdf
import pymupdf4llm
import requests
import os
from langchain.output_parsers import PydanticOutputParser


SYSTEM_PROMPT = """
You are a professor teaching a course from the following paper.
Given the contents of the paper you should output a companion document with three sections
1) A summary of the paper
2) A glossary of important terms and keywords, along with their definition and context in the paper
3) A detailed bibliography that lists the references along with a description of where in the paper they are cited and how they relate to the paper
"""

SUMMARY_PROMPT = """
You are a professor teaching a course from the following paper.
Given the contents of the paper you should output a summary of the paper
"""

KEYWORD_PROMPT = """
You are a professor teaching a course from the following paper.
Given the contents of the paper you should output a glossary of important terms and keywords, along with their definition and context in the paper
"""

CITATION_PROMPT = """
You are a professor teaching a course from the following paper.
Given the contents of the paper you should report the full list of citations along with a description of how each is used in the paper.
Each citation should only include the title and the citation number or description in the format without the authors or year information.
Also include the context of the citation in the paper, meaning how the citation is used or referenced in the paper.
"""


class ConxualizedKeyword(TypedDict):
    keyword: str
    definition: str
    local_context: str
    # global_context: str


class ConxualizedCitation(TypedDict):
    title: str
    # authors: list[str]
    # year: int
    description: str
    context: str
    # local_context: str
    # global_context: str


class ContextualizedCitationsAbstract(TypedDict):
    citation: str
    context: str
    abstract: str


class ConxualizedKeywordList(TypedDict):
    keywords: list[ConxualizedKeyword]


class ConxualizedCitationList(TypedDict):
    citations: list[ConxualizedCitation]


model_name = "llama-3.1-70b-versatile"
# model_name = "llama3-70b-8192"
# model_name = "llama-3.1-8b-instant"

model = ChatGroq(model=model_name)
keyword_model = model.with_structured_output(ConxualizedKeywordList)
summary_model = model
citation_model = model.with_structured_output(ConxualizedCitationList)


# Define the state with a built-in messages key
class ResearchState(MessagesState):
    paper_url: str
    paper_md: str
    summary: str
    keywords: ConxualizedKeywordList
    citations: ConxualizedCitationList
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
    url = state["paper_url"]
    r = requests.get(url)
    doc = pymupdf.Document(stream=r.content)
    paper = pymupdf4llm.to_markdown(doc)
    summary = summary_model.invoke(
        [
            [
                "system",
                SUMMARY_PROMPT,
            ],
            ["human", paper],
        ]
    ).content
    return {
        "paper_md": paper,
        "summary": summary,
    }


def keyword_extraction_node(state: ResearchState) -> ResearchState:
    # Logic to extract keywords
    response = keyword_model.invoke(
        [
            [
                "system",
                KEYWORD_PROMPT,
            ],
            ["human", state["paper_md"]],
        ]
    )
    return {"keywords": response}


def citation_extraction_node(state: ResearchState) -> ResearchState:
    # Logic to extract citations
    parser = PydanticOutputParser(pydantic_object=ConxualizedCitationList)
    citations = citation_model.invoke(
        [
            [
                "system",
                CITATION_PROMPT,  # + parser.get_format_instructions(),
            ],
            ["human", state["paper_md"]],
        ]
    )
    return {"citations": citations}


# def contextualization_node(state: ResearchState) -> ResearchState:
#     # Logic to add context to keywords and citations
#     return {"context": "contextualized information"}


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
# graph.add_node("contextualization_node", contextualization_node)
graph.add_node("abstract_fetching_node", abstract_fetching_node)
graph.add_node("reading_assistance_node", reading_assistance_node)
# graph.add_node("final_contextualization_node", final_contextualization_node)

# Define the edges between nodes
graph.set_entry_point("input_node")
graph.add_edge("input_node", "keyword_extraction_node")
graph.add_edge("input_node", "citation_extraction_node")
# graph.add_edge("keyword_extraction_node", "contextualization_node")
# graph.add_edge("citation_extraction_node", "contextualization_node")
graph.add_edge("citation_extraction_node", "abstract_fetching_node")
graph.add_edge("keyword_extraction_node", "reading_assistance_node")
graph.add_edge("abstract_fetching_node", "reading_assistance_node")
# graph.add_edge("reading_assistance_node", "final_contextualization_node")
graph.set_finish_point("reading_assistance_node")

# Compile the graph
app = graph.compile()
