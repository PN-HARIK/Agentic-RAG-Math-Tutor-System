import requests
import os

API_KEY = os.getenv("SERPER_API_KEY")
if not API_KEY:
    raise ValueError("Missing SERPER_API_KEY environment variable")

def web_search(query: str) -> str:
    """
    Perform a web search using Serper API.
    Returns concatenated relevant text snippets.
    """

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
    }
    params = {
        "q": query
    }

    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        
        snippets = []
        organic_results = data.get("organic", [])
        for result in organic_results:
            snippet = result.get("snippet")
            if snippet:
                snippets.append(snippet)

        if not snippets:
            return "No relevant web search results found."

        return "\n\n".join(snippets)

    except Exception as e:
        print(f"Web search error: {e}")
        return "Web search failed or no results found."
