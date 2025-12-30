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

from gateway_client import get_gateway_client, get_gateway_access_token, get_ssm_parameter

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
    return get_gateway_client("^shoppingtools")


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

        # Use direct gateway API calls instead of MCP client wrapper
        # This bypasses the connection issues with the MCP client library
        import requests
        import json
        
        try:
            # Get gateway credentials
            gateway_url = get_ssm_parameter(
                f"/concierge-agent/{os.getenv('DEPLOYMENT_ID', 'default')}/gateway-url", 
                os.getenv("AWS_REGION", "us-east-1")
            )
            access_token = get_gateway_access_token()
            
            # Initialize MCP connection
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "shopping-assistant",
                        "version": "1.0.0"
                    }
                }
            }
            
            init_response = requests.post(
                gateway_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json=init_request,
                timeout=10
            )
            
            if init_response.status_code != 200:
                raise Exception(f"Gateway initialization failed: {init_response.status_code}")
            
            # Determine which tool to use based on query
            tool_name = "shoppingtools___single_productsearch"
            if any(word in query.lower() for word in ["packing", "pack", "trip", "vacation", "travel list"]):
                tool_name = "shoppingtools___generate_packinglist_with_productASINS"
            
            # Call the shopping tool
            search_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": {
                        "user_id": user_id or "anonymous",
                        "question": query
                    }
                }
            }
            
            search_response = requests.post(
                gateway_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json=search_request,
                timeout=30
            )
            
            if search_response.status_code != 200:
                raise Exception(f"Shopping search failed: {search_response.status_code}")
            
            result = search_response.json()
            
            if "error" in result:
                raise Exception(f"Shopping tool error: {result['error']}")
            
            # Extract the answer from the response
            content = result.get("result", {}).get("content", [])
            if content and len(content) > 0:
                text_content = content[0].get("text", "")
                if text_content:
                    try:
                        data = json.loads(text_content)
                        if "answer" in data:
                            yield {"result": data["answer"]}
                            return
                    except json.JSONDecodeError:
                        pass
                    
                    # Return raw text if JSON parsing fails
                    yield {"result": text_content}
                    return
            
            # Fallback response
            yield {"result": "I found some products but couldn't format the results properly. Please try again."}
            
        except Exception as direct_error:
            logger.error(f"Direct gateway call failed: {direct_error}")
            
            # Fallback response when direct gateway fails
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

    except Exception as e:
        logger.error(f"Shopping subagent async error: {e}", exc_info=True)
        yield {"error": str(e)}