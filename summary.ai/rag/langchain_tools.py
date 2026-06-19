from typing import Optional
import requests

class GoogleSearchTool:
    name = "Google Search"
    description = "Searches Google for articles and links. Returns both most popular and latest results."

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id

    def fetch_items(self, query: str, sort: Optional[str] = None) -> list[dict]:
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": 5,
        }
        if sort == "latest":
            params["dateRestrict"] = "m1"

        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])

    def fetch_results(self, query: str, sort: Optional[str] = None) -> str:
        try:
            results = self.fetch_items(query, sort=sort)
            if not results:
                return "No results found."

            formatted = []
            for item in results:
                title = item.get("title", "No title")
                link = item.get("link", "No link")
                snippet = item.get("snippet", "")
                result_text = f"🔗 **{title}**\n{link}\n{snippet}"
                formatted.append(result_text)

            return "\n\n".join(formatted)
        except Exception as e:
            return f"Search failed: {e}"

    def _first_successful_results(self, queries: list[str], sort: Optional[str] = None) -> str:
        last_error = ""
        for query in queries:
            result = self.fetch_results(query, sort=sort)
            if result != "No results found." and not result.startswith("Search failed:"):
                return result
            last_error = result
        return last_error or "No results found."

    def _run(self, query: str, context: str = "") -> str:
        queries = [query]
        if context:
            context_terms = " ".join(context.split()[:12])
            queries.append(f"{query} {context_terms}")

        popular = self._first_successful_results(queries)
        latest = self._first_successful_results(queries, sort="latest")
        return f"### 📈 Most Popular Results:\n\n{popular}\n\n---\n\n### 🕒 Latest Results:\n\n{latest}"

    def run(self, query: str, context: str = "") -> str:
        return self._run(query, context=context)

    async def _arun(self, query: str) -> str:
        return self._run(query)
