"""
Agentic Search Tool with AutoGen GraphFlow
Iterative web search with AI-powered query refinement and result filtering
"""

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_core.models import SystemMessage, UserMessage
from autogen_core import CancellationToken
from typing import Sequence, List, Tuple, Optional, Callable, Any
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import asyncio
import inspect
import json
import ast

# SHARED CONTEXT
class SearchContext:
    """Shared context for all agents in search workflow"""
    def __init__(self):
        # User input
        self.user_query: str = ""
        self.chat_context: str = ""
        
        # Configuration
        self.max_iterations: int = 3
        self.min_results: int = 10
        self.search_category: str = "text"  # "text" or "news"
        self.ddgs_timeout: int = 20
        self.max_results_per_query: int = 15
        
        # Feature flags
        self.enable_scraping: bool = True
        self.enable_summarizing: bool = True
        
        # State
        self.iteration: int = 0
        self.accumulated_results: List[dict] = []
        self.used_queries: List[Tuple[str, str]] = []  # (query, timeline)
        self.current_iteration_results: List[dict] = []
    
    def reset(self):
        """Reset state for new search"""
        self.iteration = 0
        self.accumulated_results = []
        self.used_queries = []
        self.current_iteration_results = []

# Global shared context
search_context = SearchContext()

# AGENT 1: QUERY GENERATOR
class QueryGeneratorAgent(BaseChatAgent):
    """
    Generates optimized search queries.
    - First iteration: Returns original user query
    - Subsequent: Uses LLM to generate variations
    """
    
    def __init__(self, name: str, model_client: Optional[Any] = None):
        super().__init__(name, description="Generates search queries")
        self._model_client = model_client or OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    @property
    def produced_message_types(self) -> Sequence[type]:
        return (TextMessage,)
    
    async def _generate_query_variation(self) -> Tuple[str, str]:
        """Use LLM to generate query variation"""
        previous_queries_text = "\n".join([
            f"- Query: '{q}', Timeline: '{t}'" 
            for q, t in search_context.used_queries
        ])
        
        system_prompt = """You are a Search Query Generator.
Create one optimized search query and timeline for search engines.

Steps:
1. Consider user query + chat context + previous queries (avoid repeats)
2. Simplify: remove filler words, keep core intent
3. Assign timeline:
   - "d" → breaking/fast-changing topics
   - "w" → recent trends
   - "m" → evergreen/educational topics
4. Add content-type cue if missing (news/article/guide/tutorial)
5. Output exactly: ["<query>", "<d/w/m>"]

Example: "what are latest stock updates?" → ["latest stock market news","d"]
"""

        user_prompt = f"""Chat Context: {search_context.chat_context}

Original query: {search_context.user_query}

Previously used queries:
{previous_queries_text if previous_queries_text else "None"}

Generate a NEW variation different from previous attempts."""

        # Create messages for LLM
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt, source="user")
        ]
        
        response = await self._model_client.create(
            messages=messages,
            cancellation_token=CancellationToken()
        )
        
        try:
            response_text = response.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return (result[0], result[1])
        except:
            return (search_context.user_query, "d")
    
    async def on_messages(self, messages: Sequence, cancellation_token: CancellationToken) -> Response:
        # First iteration: use original query
        if search_context.iteration == 0:
            query = search_context.user_query
            timeline = "d"
        else:
            query, timeline = await self._generate_query_variation()
        
        search_context.used_queries.append((query, timeline))
        
        response_data = {
            "type": "search_query",
            "query": query,
            "timeline": timeline,
            "iteration": search_context.iteration
        }
        
        return Response(
            chat_message=TextMessage(
                content=json.dumps(response_data),
                source=self.name
            )
        )
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        pass


# AGENT 2: SEARCH QUERY EXECUTOR
class SearchExecutorAgent(BaseChatAgent):
    """Executes DDGS search"""
    
    def __init__(self, name: str):
        super().__init__(name, description="Executes web search")
    
    @property
    def produced_message_types(self) -> Sequence[type]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence, cancellation_token: CancellationToken) -> Response:
        # Extract query from previous message
        last_msg = messages[-1]
        data = json.loads(last_msg.content)
        query = data["query"]
        timeline = data.get("timeline", "d")
        
        # Adjust timeline for news category
        if search_context.search_category == "news" and timeline not in ["d", "w"]:
            timeline = "d"
        
        try:
            # Execute DDGS search
            with DDGS(timeout=search_context.ddgs_timeout) as ddgs:
                results = ddgs._search(
                    query=query,
                    category=search_context.search_category,
                    max_results=search_context.max_results_per_query,
                    timelimit=timeline
                )
                
                # Convert generator to list if needed
                if not isinstance(results, list):
                    results = list(results)
                
                # Store raw results in context
                search_context.current_iteration_results = results
                
                # Format results
                formatted_results = [
                    {
                        "title": result.get("title", ""),
                        "link": result.get("href", "") or result.get("link", ""),
                        "snippet": result.get("body", "")
                    }
                    for result in results
                ]
                
                response_data = {
                    "type": "search_results",
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results)
                }
                
                return Response(
                    chat_message=TextMessage(
                        content=json.dumps(response_data),
                        source=self.name
                    )
                )
        
        except Exception as e:
            # Return error
            return Response(
                chat_message=TextMessage(
                    content=json.dumps({
                        "type": "search_error",
                        "error": str(e)
                    }),
                    source=self.name
                )
            )
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        pass


# AGENT 3: REVIEWER (CUSTOM)
class ReviewerAgent(BaseChatAgent):
    """
    Reviews search results for relevance.
    Returns filtered list of relevant URLs.
    """
    
    def __init__(self, name: str, model_client: Optional[Any] = None):
        super().__init__(name, description="Reviews search results")
        self._model_client = model_client or OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    @property
    def produced_message_types(self) -> Sequence[type]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence, cancellation_token: CancellationToken) -> Response:
        # Extract search results
        last_msg = messages[-1]
        data = json.loads(last_msg.content)
        
        if data.get("type") == "search_error":
            # Pass error through
            return Response(
                chat_message=TextMessage(
                    content=last_msg.content,
                    source=self.name
                )
            )
        
        query = data["query"]
        results = data["results"]
        
        # Call LLM to review results
        system_prompt = """You are a search result reviewer.
Analyze the search results and return ONLY the indices of relevant results as a Python list.

Rules:
1. Consider the user's query and chat context
2. Return indices (1-based) of relevant results only
3. Output format: [1, 3, 5, 7] (just the list, nothing else)
4. If no results are relevant, return []
"""

        user_prompt = f"""User Query: {search_context.user_query}
Chat Context: {search_context.chat_context}

Search Query Used: {query}

Search Results:
{json.dumps(results, indent=2)}

Return relevant result indices:"""

        messages_for_llm = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt, source="user")
        ]
        
        response = await self._model_client.create(
            messages=messages_for_llm,
            cancellation_token=CancellationToken()
        )
        
        # Parse indices
        try:
            response_text = response.content.strip()
            # Remove markdown if present
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("python"):
                    response_text = response_text[6:]
                response_text = response_text.strip()
            
            result_indices = ast.literal_eval(response_text)
            
            # Filter results by indices
            relevant_results = [
                results[i - 1] 
                for i in result_indices 
                if 0 <= i - 1 < len(results)
            ]
        
        except Exception as e:
            # Fallback: return all results
            relevant_results = results
        
        response_data = {
            "type": "relevant_results",
            "results": relevant_results,
            "count": len(relevant_results)
        }
        
        return Response(
            chat_message=TextMessage(
                content=json.dumps(response_data),
                source=self.name
            )
        )
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        pass


# AGENT 4: LOOP CONTROLLER
class LoopControllerAgent(BaseChatAgent):
    """
    Controls search loop based on iteration count and result count.
    Decides whether to CONTINUE or DONE.
    """
    
    def __init__(self, name: str):
        super().__init__(name, description="Controls search loop")
    
    @property
    def produced_message_types(self) -> Sequence[type]:
        return (TextMessage,)
    
    def _deduplicate_results(self, results: List[dict]) -> List[dict]:
        """Remove duplicate URLs"""
        seen_links = set()
        deduplicated = []
        
        for result in results:
            link = result.get("link")
            if link and link not in seen_links:
                seen_links.add(link)
                deduplicated.append(result)
        
        return deduplicated
    
    async def on_messages(self, messages: Sequence, cancellation_token: CancellationToken) -> Response:
        # Extract relevant results
        last_msg = messages[-1]
        data = json.loads(last_msg.content)
        
        if data.get("type") == "search_error":
            # Search failed, return DONE with error
            return Response(
                chat_message=TextMessage(
                    content=json.dumps({
                        "type": "DONE",
                        "error": data["error"],
                        "accumulated_results": search_context.accumulated_results
                    }),
                    source=self.name
                )
            )
        
        new_results = data["results"]
        
        # Accumulate results
        search_context.accumulated_results.extend(new_results)
        
        # Deduplicate
        search_context.accumulated_results = self._deduplicate_results(
            search_context.accumulated_results
        )
        
        # Increment iteration
        search_context.iteration += 1
        
        # Decision logic
        if (search_context.iteration >= search_context.max_iterations or 
            len(search_context.accumulated_results) >= search_context.min_results):
            # DONE
            response_data = {
                "type": "DONE",
                "accumulated_results": search_context.accumulated_results,
                "total_count": len(search_context.accumulated_results),
                "iterations_used": search_context.iteration
            }
        else:
            # CONTINUE
            response_data = {
                "type": "CONTINUE",
                "current_count": len(search_context.accumulated_results),
                "iteration": search_context.iteration
            }
        
        return Response(
            chat_message=TextMessage(
                content=json.dumps(response_data),
                source=self.name
            )
        )
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        pass


# AGENT 5: DATA PROCESSING (scraping + summarization)
class DataProcessorAgent(BaseChatAgent):
    """
    Scrapes and summarizes content from URLs.
    Scraping runs in parallel for maximum speed.
    Summarization runs as single batch operation.
    """
    
    def __init__(
        self, 
        name: str,
        scraper_function: Optional[Callable[[str], str]] = None,
        summarizer_function: Optional[Callable[[List[str]], List[str]]] = None
    ):
        super().__init__(name, description="Scrapes and summarizes content")
        self._scraper = scraper_function
        self._summarizer = summarizer_function
    
    @property
    def produced_message_types(self) -> Sequence[type]:
        return (TextMessage,)
    
    @staticmethod
    def get_favicon_url(base_url: str) -> str:
        """
        Generate the favicon URL for a given base URL.

        Args:
            base_url: The base URL of the website.

        Returns:
            The URL of the website's favicon.
        """
        parsed_url = urlparse(base_url)
        favicon_url = f"https://{parsed_url.netloc}/favicon.ico"
        return favicon_url
    
    async def on_messages(self, messages: Sequence, cancellation_token: CancellationToken) -> Response:
        # Extract accumulated results
        last_msg = messages[-1]
        data = json.loads(last_msg.content)
        
        if data.get("error"):
            # Return error
            return Response(
                chat_message=TextMessage(
                    content=json.dumps({
                        "error": data["error"],
                        "organic": []
                    }),
                    source=self.name
                )
            )
        
        results = data["accumulated_results"]
        
        # Always add favicon to each result
        for result in results:
            if result.get("link"):
                result["fav_icon"] = self.get_favicon_url(result["link"])
        
        # Step 1: Conditional scraping
        scraped_texts = []
        if search_context.enable_scraping and self._scraper:
            # Scrape all URLs in parallel (sync scraper with ThreadPoolExecutor)
            with ThreadPoolExecutor(max_workers=24) as executor:
                loop = asyncio.get_event_loop()
                scrape_tasks = [
                    loop.run_in_executor(executor, self._scraper, result.get("link"))
                    for result in results
                    if result.get("link")
                ]
                scraped_texts = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        else:
            # Scraping disabled - create empty list
            scraped_texts = [None] * len(results)
        
        # Step 2: Filter out errors and prepare list for batch summarization
        valid_scraped_texts = []
        valid_indices = []
        scraped_data_map = {}  # Map index to scraped text
        
        if search_context.enable_scraping:
            for i, scraped_text in enumerate(scraped_texts):
                if not isinstance(scraped_text, Exception) and scraped_text:
                    valid_scraped_texts.append(scraped_text)
                    valid_indices.append(i)
                    scraped_data_map[i] = scraped_text
                elif isinstance(scraped_text, Exception):
                    scraped_data_map[i] = f"Error: {str(scraped_text)}"
        
        # Step 3: Conditional batch summarization
        summaries = []
        summary_map = {}  # Map index to summary
        
        if search_context.enable_scraping and search_context.enable_summarizing and valid_scraped_texts and self._summarizer:
            summaries_result = self._summarizer(valid_scraped_texts)  # Pass list, get list back
            # print("summarizer running")
            # Check if result is awaitable (async function)
            if inspect.iscoroutine(summaries_result):
                summaries = await summaries_result
            else:
                summaries = summaries_result
            
            # Map summaries back to indices
            for idx, summary_idx in enumerate(valid_indices):
                if idx < len(summaries):
                    summary_map[summary_idx] = summaries[idx]
        
        # Step 4: Build final results with proper ordering
        processed_results = []
        for i, result in enumerate(results):
            if not result.get("link"):
                continue
            
            # Base result structure
            processed_result = {
                "title": result.get("title", ""),
                "link": result["link"],
                "snippet": result.get("snippet", ""),
                "fav_icon": result.get("fav_icon", "")
            }
            
            # Add scrapped_data or summarized_data based on flags
            if search_context.enable_scraping:
                if search_context.enable_summarizing and i in summary_map:
                    # Summarization enabled - add summarized_data
                    processed_result["summarized_data"] = summary_map[i]
                elif i in scraped_data_map:
                    # Only scraping enabled - add scrapped_data
                    processed_result["scrapped_data"] = scraped_data_map[i]
            
            processed_results.append(processed_result)
        
        # Final output in exact format
        final_output = {
            "organic": processed_results,
            "query": search_context.user_query,
            "context": search_context.chat_context,
            "total_results": len(processed_results)
        }
        print("\n","*"*20,"final output by the search tool agent","*"*20)
        print("\n",final_output,"\n")
        
        return Response(
            chat_message=TextMessage(
                content=json.dumps(final_output),
                source=self.name
            )
        )
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        pass


# GRAPHFLOW MANAGER
class SearchGraphFlowManager:
    """
    Manages the search GraphFlow.
    Build once, reuse for all searches.
    """
    
    def __init__(
        self,
        scraper_function: Callable[[str], str],
        summarizer_function: Callable[[str], str],
        model_client: Optional[Any] = None
    ):
        """
        Initialize the search workflow.
        
        Args:
            scraper_function: Function that takes URL and returns raw text
            summarizer_function: Function that takes text and returns summary
            model_client: LLM client for query generation and review
        """
        self.model_client = model_client or OpenAIChatCompletionClient(model="gpt-4o-mini")
        
        # Create agents
        self.query_generator = QueryGeneratorAgent("query_generator", self.model_client)
        self.search_executor = SearchExecutorAgent("search_executor")
        self.reviewer = ReviewerAgent("reviewer", self.model_client)
        self.loop_controller = LoopControllerAgent("loop_controller")
        self.data_processor = DataProcessorAgent(
            "data_processor",
            scraper_function,
            summarizer_function
        )
        
        # Build graph
        builder = DiGraphBuilder()
        builder.add_node(self.query_generator)
        builder.add_node(self.search_executor)
        builder.add_node(self.reviewer)
        builder.add_node(self.loop_controller)
        builder.add_node(self.data_processor)
        
        # Edges
        builder.add_edge(self.query_generator, self.search_executor)
        builder.add_edge(self.search_executor, self.reviewer)
        builder.add_edge(self.reviewer, self.loop_controller)
        
        # Loop back on CONTINUE
        builder.add_edge(
            self.loop_controller,
            self.query_generator,
            condition=lambda msg: "CONTINUE" in msg.content
        )
        
        # Exit to DataProcessor on DONE
        builder.add_edge(
            self.loop_controller,
            self.data_processor,
            condition=lambda msg: "DONE" in msg.content
        )
        
        builder.set_entry_point(self.query_generator)
        
        # Build flow
        self.graph = builder.build()
        self.flow = GraphFlow(
            participants=[
                self.query_generator,
                self.search_executor,
                self.reviewer,
                self.loop_controller,
                self.data_processor
            ],
            graph=self.graph,
            max_turns=10
        )
    
    async def search(
        self,
        user_query: str,
        chat_context: str = "",
        max_iterations: int = 3,
        min_results: int = 10,
        search_category: str = "text",
        enable_scraping: bool = True,
        enable_summarizing: bool = True
    ) -> str:
        """
        Execute search workflow.
        
        Args:
            user_query: User's search query
            chat_context: Chat conversation context
            max_iterations: Maximum search iterations
            min_results: Minimum required results
            search_category: "text" or "news"
            enable_scraping: Enable scraping of URLs (default: True)
            enable_summarizing: Enable summarization (requires scraping) (default: True)
        
        Returns:
            JSON string with results
        """
        # Validation: scraping OFF forces summarizing OFF
        if not enable_scraping:
            enable_summarizing = False
        
        # Update shared context
        search_context.user_query = user_query
        search_context.chat_context = chat_context
        search_context.max_iterations = max_iterations
        search_context.min_results = min_results
        search_context.search_category = search_category
        search_context.enable_scraping = enable_scraping
        search_context.enable_summarizing = enable_summarizing
        search_context.reset()
        
        # Run flow
        result = await self.flow.run(task="START")
        
        # Return final message content
        return result.messages[-1].content


# TOOL WRAPPER FOR CHATBOT
def create_search_tool(
    scraper_function: Callable[[str], str],
    summarizer_function: Callable[[str], str],
    model_client: Optional[Any] = None
) -> FunctionTool:
    """
    Create search tool for chatbot integration.
    
    Args:
        scraper_function: URL scraping function (url: str -> raw_text: str)
        summarizer_function: Text summarization function (text: str -> summary: str)
        model_client: LLM client for query generation and review
    
    Returns:
        FunctionTool that can be added to chatbot's tools list
    """
    # Initialize manager once
    manager = SearchGraphFlowManager(
        scraper_function=scraper_function,
        summarizer_function=summarizer_function,
        model_client=model_client
    )
    
    async def agentic_search(
        user_query: str,
        chat_context: str = "",
        max_iterations: int = 3,
        min_results: int = 10
    ) -> str:
        """
        Perform agentic web search with iterative query refinement and AI filtering.
        
        This tool searches the web, filters results using AI, and returns summarized content.
        It automatically refines queries if initial results are insufficient.
        
        Args:
            user_query: The search query extracted from user's question.
                       Should be concise keywords or phrases.
                       Example: "latest quantum computing breakthroughs"
            
            chat_context: Brief 1-2 sentence summary of the conversation context.
                         Include: what user is interested in, previous topics discussed.
                         Example: "User previously asked about AI, now interested in quantum tech"
                         Use empty string if no prior context exists.
            
            max_iterations: Maximum number of search attempts with query refinement (default: 3)
            
            min_results: Minimum number of quality results to collect (default: 10)
        
        Returns:
            JSON string containing:
            {
                "results": [
                    {
                        "url": "...",
                        "title": "...",
                        "summary": "..."
                    }
                ],
                "total": <count>,
                "query": "<original query>"
            }
        """
        return await manager.search(
            user_query=user_query,
            chat_context=chat_context,
            max_iterations=max_iterations,
            min_results=min_results
        )
    description = agentic_search.__doc__
    return FunctionTool(
        agentic_search,
        # description="Search the web with AI-powered query refinement, result filtering, and content summarization"
        description = description
    )


# USAGE EXAMPLE
from summarizer import summarize_articles
from web_scraper import scrape_website
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAIChatCompletionClient(
    model='gpt-4.1-mini',
    api_key=os.getenv("OPENAI_API_KEY")
)
search_tool = create_search_tool(
    scraper_function=scrape_website,
    summarizer_function=summarize_articles,
    model_client=client
)
zenith_chatbot = AssistantAgent(
    name = 'chatbot',
    model_client=client,
    tools = [search_tool],
    reflect_on_tool_use=True,
    system_message="""You are a helpful assistant with web search capabilities.
How to use the tool:
1. user_query: Extract core search keywords from user's question
   - Remove filler words like "what are", "tell me about"
   - Example: "What are the latest AI developments?" → "latest AI developments"

2. chat_context: Provide 1-2 sentence conversation summary
   - What topic is user interested in?
   - What have they asked before?
   - Example: "User previously asked about machine learning, now interested in recent AI news"
   - If first question, use empty string ""

3. Use default values for max_iterations and min_results unless user specifies"""
    )
async def main():
    response = await zenith_chatbot.run(task="Latest stock market news and updates?")
    # response = await Console(zenith_chatbot.run_stream(task="What are the latest breakthroughs in quantum computing?"))
    # response = await zenith_chatbot.run(task="What are the latest breakthroughs in quantum computing?")
    print(response.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())

# pass this "search_tool" in the tools list of zenith chatbot
# below is the recommended system prompt for configuring the zenith to generate relevant chat_context and search_query 
"""
# Configure zenith chatbot with recommended system prompt

system_message='''You are a helpful assistant with web search capabilities.

.......

How to use the tool:
1. user_query: Extract core search keywords from user's question
   - Remove filler words like "what are", "tell me about"
   - Example: "What are the latest AI developments?" → "latest AI developments"

2. chat_context: Provide 1-2 sentence conversation summary
   - What topic is user interested in?
   - What have they asked before?
   - Example: "User previously asked about machine learning, now interested in recent AI news"
   - If first question, use empty string ""

3. Use default values for max_iterations and min_results unless user specifies

.......
"""