# LLM Zoomcamp HW1 - Agentic RAG

import os

from dotenv import load_dotenv
from openai import OpenAI
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index

from rag import RAG

load_dotenv()

# openrouter is openai-compatible, just point at their url
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "openrouter/free"  # or qwen/qwen3-coder:free, meta-llama/llama-3.3-70b-instruct:free

QUERY = "How does the agentic loop keep calling the model until it stops?"

# pull the lessons at the pinned commit so everyone has the same data
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [f.parse() for f in reader.read()]

# Q1
print("Q1 (lesson pages):", len(documents))

# Q2
index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)
print("Q2 (first result):", index.search(QUERY, num_results=5)[0]["filename"])

# Q3
rag_full = RAG(index, client, MODEL)
_, usage = rag_full.rag(QUERY)
print("Q3 (input tokens):", usage.prompt_tokens)

# Q4
chunks = chunk_documents(documents, size=2000, step=1000)
print("Q4 (chunks):", len(chunks))

# Q5 - same query, but over the smaller chunks
chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)
rag_chunk = RAG(chunk_index, client, MODEL)
_, usage_c = rag_chunk.rag(QUERY)
print("Q5 (input tokens):", usage_c.prompt_tokens,
      "| reduction: %.1fx" % (usage.prompt_tokens / usage_c.prompt_tokens))

# Q6 - give the model a search tool and let it decide when to call it
from toyaikit.tools import Tools
from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.chat.runners import OpenAIChatCompletionsRunner
from toyaikit.chat import IPythonChatInterface

search_calls = 0

def search(query: str) -> list:
    """Search the course lessons for passages relevant to the query.

    Args:
        query: what to look for in the course material.
    """
    global search_calls
    search_calls += 1
    return chunk_index.search(query, num_results=5)

tools = Tools()
tools.add_tool(search)

AGENT_INSTRUCTIONS = (
    "You're a course teaching assistant. Answer the student's question using "
    "the search tool. Make multiple searches with different keywords before answering."
)

# chat-completions client/runner - openrouter doesn't do the responses api
llm_client = OpenAIChatCompletionsClient(model=MODEL, client=client)
runner = OpenAIChatCompletionsRunner(
    tools=tools,
    developer_prompt=AGENT_INSTRUCTIONS,
    chat_interface=IPythonChatInterface(),
    llm_client=llm_client,
)

runner.loop("How does the agentic loop work, and how is it different from plain RAG?")
print("Q6 (search calls):", search_calls)