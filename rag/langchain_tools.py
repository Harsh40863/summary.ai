from langchain.tools import BaseTool
from typing import Optional
import requests

class GoogleSearchTool(BaseTool):
    name: str = "Google Search"
    description: str = "Searches Google for articles and links. Returns both most popular and latest results."

    api_key: str
    cse_id: str

    def fetch_results(self, query: str, sort: Optional[str] = None) -> str:
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": 5,
        }
        if sort == "latest":
            params["sort"] = "date"

        try:
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("items", [])
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

    def _run(self, query: str) -> str:
        popular = self.fetch_results(query)
        latest = self.fetch_results(query, sort="latest")
        return f"### 📈 Most Popular Results:\n\n{popular}\n\n---\n\n### 🕒 Latest Results:\n\n{latest}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
