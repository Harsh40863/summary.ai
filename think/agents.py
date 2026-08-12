from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from .tools import web_search, scrape_url
from langchain_core.prompts import ChatPromptTemplate

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

model_name = os.getenv("MISTRAL_MODEL", "mistral-medium-3-5")
api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model=model_name,
    api_key=api_key,
    temperature=0
)

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )

#writter chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""
    )
])

writter_chain=writer_prompt|llm|StrOutputParser()

#critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research critic. Carefully evaluate the research report for factual accuracy, completeness, clarity, logical consistency, and quality of sources."
    ),
    (
        "human",
        """Critically review the research report below.

Topic: {topic}

Research Report:
{report}

Evaluate the report for:
- Factual accuracy
- Missing or important information
- Logical consistency
- Quality and reliability of sources
- Clarity and structure
- Unsupported or potentially misleading claims

Provide specific corrections and improvements where necessary.

Be detailed, objective and professional."""
    )
])


critic_chain=critic_prompt|llm|StrOutputParser()
