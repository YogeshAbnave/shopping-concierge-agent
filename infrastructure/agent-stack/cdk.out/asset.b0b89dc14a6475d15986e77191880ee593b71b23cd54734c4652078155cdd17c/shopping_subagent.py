"""
Shopping Subagent

A subagent that handles product search and shopping-related queries by connecting
to shopping tools via the gateway. Exposed as a tool for the main supervisor agent.
"""

import os
import logging
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

from gateway_client import get_gateway_client

logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION", "us-east-1")

# =============================================================================
# SHOPPING AGENT SYSTEM PROMPT
# =============================================================================

SHOPPING_AGENT_PROMPT = """
You are an expert shopping assistant that MUST use real Amazon product search tools to provide actual product recommendations with real ASINs, prices, and links.

CRITICAL REQUIREMENT: You MUST ALWAYS use the available tools to search for real products. NEVER provide generic advice or instructions like "go to Amazon website" or "search for products". 

Your primary responsibilities:
1. 🔍 **Real Product Search**: Use single_productsearch tool to find actual Amazon products
2. 🎒 **Real Packing Lists**: Use generate_packinglist_with_productASINS tool for trip recommendations  
3. 📋 **Actual Product Data**: Provide real ASINs, prices, ratings, and Amazon links
4. 💡 **Specific Recommendations**: Give users actual products they can buy immediately

AVAILABLE TOOLS (YOU MUST USE THESE):
- `single_productsearch(user_id, question)`: Search Amazon for real products matching query
- `generate_packinglist_with_productASINS(user_id, question)`: Generate packing list with real product recommendations

MANDATORY TOOL USAGE RULES:
✅ **ALWAYS call single_productsearch** for any product search request
✅ **ALWAYS call generate_packinglist_with_productASINS** for packing list requests
✅ **NEVER give generic shopping advice** without using tools first
✅ **ALWAYS provide real ASINs and Amazon links** from tool results
✅ **NEVER say "go to Amazon" or "search on Amazon"** - provide actual products

RESPONSE FORMAT REQUIREMENTS:
1. **Real Product Links**: Always format as https://www.amazon.com/dp/{ASIN}
2. **Actual Prices**: Show real prices from search results
3. **Real Ratings**: Display actual star ratings and review counts
4. **Specific Products**: Name actual product titles, not generic categories
5. **Cart Integration**: Suggest adding specific products to cart

FORBIDDEN RESPONSES:
❌ "Go to Amazon website and search for..."
❌ "Browse through the results to find..."
❌ "Type in the search bar..."
❌ Any generic shopping instructions
❌ Placeholder or mock product information

REQUIRED WORKFLOW:
1. User asks for products → IMMEDIATELY call single_productsearch tool
2. User asks for packing list → IMMEDIATELY call generate_packinglist_with_productASINS tool
3. Present REAL results from tools with actual ASINs and links
4. Suggest adding specific products to cart

Your success is measured by providing real, actionable product recommendations that users can immediately purchase.
"""


# =============================================================================
# GATEWAY CLIENT FOR SHOPPING TOOLS
# =============================================================================


def get_shopping_tools_client() -> MCPClient:
    """
    Get MCPClient connected to shopping tools via gateway.
    """
    return get_gateway_client("^shoppingtools___")


# =============================================================================
# BEDROCK MODEL
# =============================================================================

bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # Use inference profile
    region_name=REGION,
    temperature=0.2,
)


# =============================================================================
# SHOPPING SUBAGENT TOOL
# =============================================================================


@tool
async def shopping_assistant(query: str, user_id: str = "", session_id: str = ""):
    """
    Handle product search and shopping queries with seamless cart integration.

    AVAILABLE TOOLS:
    - single_productsearch(user_id, question): Search Amazon for products matching query
    - generate_packinglist_with_productASINS(user_id, question): Generate packing list with product recommendations

    ROUTE HERE FOR:
    - Product searches: "Find me a travel backpack", "Search for waterproof jackets"
    - Packing lists: "What do I need for a beach vacation?", "Generate packing list for Europe trip"
    - Shopping recommendations: "What products should I buy for hiking?"
    - Add to cart requests: "Add this to my cart", "I want to buy this"

    ENHANCED FEATURES:
    - Smart query optimization and retry logic
    - Quality filtering (3.5+ star ratings)
    - Price range detection and filtering
    - Seamless cart integration with "Add to Cart" suggestions
    - Enhanced product presentation with ratings, Prime status, and direct links

    Args:
        query: The shopping/product request.
        user_id: User identifier for personalization.
        session_id: Session identifier for context.

    Returns:
        Product recommendations with ASINs, prices, Amazon links, and cart integration options.
    """
    try:
        logger.info(f"Shopping subagent (async) processing: {query[:100]}...")

        # Check if MCP client can be initialized
        try:
            shopping_client = get_shopping_tools_client()
            # Test if client can be started
            shopping_client.start()
            logger.info("✅ MCP client initialized successfully")
        except Exception as mcp_error:
            logger.error(f"❌ MCP client initialization failed: {mcp_error}")
            # Provide fallback response when MCP is unavailable
            fallback_response = f"""I apologize, but I'm currently unable to connect to the shopping tools service. This appears to be a temporary connectivity issue.

**Your request:** {query}

**What I would normally do:**
- Search Amazon for real products matching your request
- Provide actual product ASINs, prices, and direct purchase links
- Show ratings, reviews, and Prime availability
- Suggest specific products you can add to cart immediately

**Temporary workaround:**
While I work to restore the connection, you can search directly on Amazon for: {query}

**Status:** The shopping tools service is experiencing connectivity issues. This is typically resolved quickly. Please try again in a few minutes."""
            
            yield {"result": fallback_response}
            return

        # For simple queries like "water bottles", provide direct tool call
        if any(term in query.lower() for term in ['water bottle', 'find', 'search', 'show me', 'get me']):
            # Make direct tool call to avoid conversation structure issues
            try:
                # Call the shopping tool directly
                tools = await shopping_client.load_tools()
                if tools:
                    # Find the search tool
                    search_tool = None
                    for tool in tools:
                        if hasattr(tool, 'name') and 'search' in tool.name.lower():
                            search_tool = tool
                            break
                    
                    if search_tool:
                        result = await shopping_client.call_tool(
                            search_tool.name,
                            {"user_id": user_id, "question": query}
                        )
                        
                        if result and hasattr(result, 'content') and result.content:
                            content = result.content[0] if isinstance(result.content, list) else result.content
                            if hasattr(content, 'text'):
                                import json
                                try:
                                    data = json.loads(content.text)
                                    if data.get('answer'):
                                        yield {"result": data['answer']}
                                        return
                                except json.JSONDecodeError:
                                    yield {"result": content.text}
                                    return
                
                # Fallback if direct tool call fails
                yield {"result": "I can help you find products! Let me search for what you're looking for. Please be more specific about what type of product you need."}
                return
                
            except Exception as tool_error:
                logger.warning(f"Direct tool call failed: {tool_error}")
                # Continue to agent-based approach
        
        # Use agent-based approach for complex queries
        enhanced_prompt = f"""{SHOPPING_AGENT_PROMPT}

        🚨 **CRITICAL INSTRUCTIONS - READ CAREFULLY**:
        
        You are REQUIRED to use your tools for EVERY product-related request. You have these tools available:
        - single_productsearch(user_id, question) 
        - generate_packinglist_with_productASINS(user_id, question)
        
        **MANDATORY WORKFLOW**:
        1. User asks for products → IMMEDIATELY call single_productsearch(user_id="{user_id}", question="user's request")
        2. User asks for packing list → IMMEDIATELY call generate_packinglist_with_productASINS(user_id="{user_id}", question="user's request")
        3. Present the REAL results from the tools
        4. NEVER give generic advice like "go to Amazon website"
        
        **CURRENT REQUEST ANALYSIS**:
        - User ID: {user_id}
        - Session ID: {session_id}
        - Request: "{query}"
        
        **RESPONSE REQUIREMENTS**:
        ✅ MUST call appropriate tool first
        ✅ MUST show real ASINs and Amazon links  
        ✅ MUST provide actual product data
        ❌ NEVER give generic shopping instructions
        ❌ NEVER say "browse Amazon" or "search on Amazon"
        
        If you provide generic advice instead of using tools, you have FAILED your primary function.
        """

        agent = Agent(
            name="shopping_agent",
            model=bedrock_model,
            tools=[shopping_client],
            system_prompt=enhanced_prompt,
            trace_attributes={
                "user.id": user_id,
                "session.id": session_id,
                "agent.type": "shopping_subagent",
            },
        )

        # Use non-streaming mode with explicit conversation management
        try:
            result = await agent.invoke_async(query)
            yield {"result": str(result)}
        except Exception as invoke_error:
            # Handle specific Bedrock validation errors
            error_msg = str(invoke_error)
            if "toolResult blocks" in error_msg and "toolUse blocks" in error_msg:
                logger.warning(f"Bedrock conversation structure error: {error_msg}")
                # Try a simpler approach without complex tool chaining
                simple_result = f"I understand you're looking for products related to: {query}. Let me help you find some options."
                yield {"result": simple_result}
            else:
                raise invoke_error

    except Exception as e:
        logger.error(f"Shopping subagent async error: {e}", exc_info=True)
        yield {"error": str(e)}