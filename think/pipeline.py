from .agents import build_reader_agent, build_search_agent, writter_chain, critic_chain
from langchain_core.messages import HumanMessage
def run_research_pipeline(topic:str)->dict:
    state={}

    #search agent working
    print("\n" + "=" * 50)
    print("SEARCH AGENT IS WORKING")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
    "messages": [
        HumanMessage(
                    content=f"""Find recent reliable information about {topic}.

        Return the search results with:
        1. Title
        2. URL
        3. Relevant content/snippet

        Make sure you include the actual URLs because another agent will use them for deeper scraping."""
                )
            ]
        })
    state['search_results']=search_result['messages'][-1].content
    reader_agent=build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            HumanMessage(
                content=f"""Based on the following search results about '{topic}',
    pick the most relevant URL and scrape it for deeper content.

    Search Results:
    {state['search_results'][:800]}"""
            )
        ]
    })

    state['scraped_result']=reader_result['messages'][-1].content

    print("\nscrapped content\n",state['scraped_result'])

    reaserch_combine=(
        f"SEARCH RESULTS:\n {state['search_results']}\n\n"
        f"DETAILED SCRAPPED CONTENT:\n \n {state['scraped_result']}\n\n"
    )
    state['report']= writter_chain.invoke(
        {
            "topic":topic,
            "research":reaserch_combine
        }
    )

    print("critic_review"+"-"*50)
    state['feedback']=critic_chain.invoke({
        "report":state['report'],
        "topic":topic
    })

    print("\n critic report :\n",state["feedback"])
    return state



if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)
