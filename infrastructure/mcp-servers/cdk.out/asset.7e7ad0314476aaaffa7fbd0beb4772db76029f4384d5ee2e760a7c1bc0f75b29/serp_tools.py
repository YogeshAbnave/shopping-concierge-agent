import os
import logging
import boto3
import re
from typing import Any, Dict, List
from serpapi import GoogleSearch

logger = logging.getLogger(__name__)


def get_ssm_parameter(parameter_name: str, region: str) -> str:
    """
    Fetch parameter from SSM Parameter Store.

    Args:
        parameter_name: SSM parameter name
        region: AWS region

    Returns:
        Parameter value
    """
    ssm = boto3.client("ssm", region_name=region)
    try:
        response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except ssm.exceptions.ParameterNotFound:
        raise ValueError(f"SSM parameter not found: {parameter_name}")
    except Exception as e:
        raise ValueError(f"Failed to retrieve SSM parameter {parameter_name}: {e}")


def get_serpapi_key() -> str:
    """
    Get SerpAPI key from AWS SSM Parameter Store.

    Returns:
        SerpAPI key
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    return get_ssm_parameter("/concierge-agent/shopping/serp-api-key", region)


def optimize_search_query(query: str) -> str:
    """
    Optimize user query for better Amazon search results.
    
    Args:
        query: Original user query
        
    Returns:
        Optimized search query
    """
    # Convert to lowercase for processing
    query_lower = query.lower()
    
    # Remove common filler words that don't help search
    filler_words = ['i need', 'i want', 'looking for', 'find me', 'search for', 'show me', 'get me']
    for filler in filler_words:
        query_lower = query_lower.replace(filler, '').strip()
    
    # Add specific keywords for better results
    improvements = {
        'hiking': 'hiking outdoor gear',
        'travel': 'travel accessories',
        'beach': 'beach vacation',
        'winter': 'winter gear',
        'workout': 'fitness exercise',
        'office': 'office supplies',
        'kitchen': 'kitchen appliances',
        'phone': 'smartphone accessories',
        'laptop': 'laptop computer',
        'camera': 'digital camera photography'
    }
    
    for keyword, improvement in improvements.items():
        if keyword in query_lower and improvement not in query_lower:
            query_lower = f"{improvement} {query_lower}"
            break
    
    # Clean up extra spaces
    query_optimized = ' '.join(query_lower.split())
    
    logger.info(f"Query optimization: '{query}' -> '{query_optimized}'")
    return query_optimized


def extract_price_range(query: str) -> tuple:
    """
    Extract price range from query if mentioned.
    
    Args:
        query: Search query
        
    Returns:
        Tuple of (min_price, max_price) or (None, None)
    """
    # Look for price patterns like "under $50", "$20-$100", "between $10 and $30"
    price_patterns = [
        r'under\s*\$?(\d+)',
        r'below\s*\$?(\d+)',
        r'less than\s*\$?(\d+)',
        r'\$?(\d+)\s*-\s*\$?(\d+)',
        r'between\s*\$?(\d+)\s*and\s*\$?(\d+)',
        r'\$?(\d+)\s*to\s*\$?(\d+)'
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, query.lower())
        if match:
            if 'under' in pattern or 'below' in pattern or 'less than' in pattern:
                return (None, int(match.group(1)))
            else:
                return (int(match.group(1)), int(match.group(2)))
    
    return (None, None)


def search_amazon_products(query: str, max_results: int = 15, min_rating: float = 3.5) -> Dict[str, Any]:
    """
    Search for products on Amazon using SerpAPI with enhanced parameters.

    Args:
        query: Search query for products
        max_results: Maximum number of results to return
        min_rating: Minimum rating filter

    Returns:
        Dict containing search results with product information
    """
    try:
        api_key = get_serpapi_key()
        
        # Optimize the search query
        optimized_query = optimize_search_query(query)
        
        # Extract price range if mentioned
        min_price, max_price = extract_price_range(query)

        # Enhanced search parameters
        params = {
            "engine": "amazon",
            "amazon_domain": "amazon.com",
            "k": optimized_query,
            "api_key": api_key,
        }
        
        # Add price filters if detected
        if min_price:
            params["low_price"] = min_price
        if max_price:
            params["high_price"] = max_price

        search = GoogleSearch(params)
        results = search.get_dict()

        # Extract and filter product information
        products = []
        organic_results = results.get("organic_results", [])

        for product in organic_results:
            # Extract rating as float
            rating = 0
            if product.get("rating"):
                try:
                    rating = float(product.get("rating", 0))
                except (ValueError, TypeError):
                    rating = 0
            
            # Filter by minimum rating
            if rating > 0 and rating < min_rating:
                continue
                
            # Extract price more robustly
            price_value = "N/A"
            price_raw = product.get("price")
            if isinstance(price_raw, dict):
                price_value = price_raw.get("value", "N/A")
            elif isinstance(price_raw, str):
                # Extract numeric value from price string
                price_match = re.search(r'\$?([\d,]+\.?\d*)', price_raw)
                if price_match:
                    price_value = float(price_match.group(1).replace(',', ''))
                else:
                    price_value = price_raw
            elif isinstance(price_raw, (int, float)):
                price_value = price_raw

            product_info = {
                "asin": product.get("asin", ""),
                "title": product.get("title", ""),
                "link": product.get("link", ""),
                "price": price_value,
                "rating": rating,
                "reviews": product.get("reviews", 0),
                "thumbnail": product.get("thumbnail", ""),
                "delivery": product.get("delivery", ""),
                "prime": product.get("prime", False),
                "sponsored": product.get("sponsored", False)
            }
            
            # Only add products with ASIN (real products)
            if product_info["asin"]:
                products.append(product_info)
                
            # Stop when we have enough results
            if len(products) >= max_results:
                break

        # Sort by rating and reviews for better quality
        products.sort(key=lambda x: (x["rating"], x["reviews"]), reverse=True)

        return {"success": True, "products": products, "total_results": len(products)}

    except Exception as e:
        logger.error(f"Error searching Amazon products: {e}")
        return {"success": False, "error": str(e), "products": [], "total_results": 0}


def search_products(user_id: str, question: str) -> Dict[str, Any]:
    """
    Process a product search request from user by searching products on Amazon via SerpAPI.
    Includes retry logic and query refinement for better results.

    Args:
        user_id: The unique identifier of the user for whom products are being searched.
        question: User's query text requesting product information

    Returns:
        Dict: A dictionary called 'product_list' with search results
            - 'answer': Description of found products or error message
            - 'asins': List of ASINs found
            - 'products': List of product details
    """
    try:
        logger.info(f"Processing product search for user {user_id}: {question}")

        # Try multiple search strategies
        search_attempts = [
            question,  # Original query
            optimize_search_query(question),  # Optimized query
        ]
        
        # Add more specific variations based on query content
        question_lower = question.lower()
        if any(word in question_lower for word in ['best', 'top', 'good', 'recommend']):
            search_attempts.append(f"best rated {optimize_search_query(question)}")
        
        if 'cheap' in question_lower or 'budget' in question_lower:
            search_attempts.append(f"budget {optimize_search_query(question)}")

        products = []
        last_error = None
        
        # Try each search strategy
        for attempt, search_query in enumerate(search_attempts, 1):
            logger.info(f"Search attempt {attempt}: {search_query}")
            
            search_results = search_amazon_products(search_query, max_results=12)
            
            if search_results["success"] and search_results["products"]:
                products = search_results["products"]
                logger.info(f"Found {len(products)} products on attempt {attempt}")
                break
            else:
                last_error = search_results.get('error', 'No products found')
                logger.warning(f"Attempt {attempt} failed: {last_error}")

        if not products:
            return {
                "answer": f"No products found matching '{question}'. Try being more specific or using different keywords. Error: {last_error}",
                "asins": [],
                "products": [],
            }

        # Extract ASINs
        asins = [p["asin"] for p in products if p.get("asin")]

        # Build enhanced response with better formatting and cart integration
        answer = f"🛍️ **Found {len(products)} high-quality products for '{question}'**\n\n"
        
        for i, product in enumerate(products[:8], 1):  # Show top 8 results
            # Format price
            if isinstance(product["price"], (int, float)):
                price_str = f"${product['price']:.2f}"
            else:
                price_str = str(product["price"])
            
            # Build product entry with enhanced formatting
            answer += f"**{i}. {product['title'][:80]}{'...' if len(product['title']) > 80 else ''}**\n"
            answer += f"   💰 **Price**: {price_str}"
            
            if product.get("rating") and product["rating"] > 0:
                stars = "⭐" * int(product["rating"])
                answer += f" | {stars} {product['rating']}/5"
                if product.get("reviews"):
                    answer += f" ({product['reviews']} reviews)"
            
            if product.get("prime"):
                answer += " | 🚚 **Prime Eligible**"
                
            answer += f"\n   🔗 **ASIN**: `{product['asin']}`"
            answer += f"\n   🛒 **Link**: https://www.amazon.com/dp/{product['asin']}"
            
            # Add cart-ready data for easy integration
            answer += f"\n   📦 **Cart Data**: ASIN: {product['asin']}, Price: {price_str}\n\n"

        # Enhanced footer with clear cart integration
        answer += "🎯 **Ready to Shop?**\n"
        answer += "• Tell me which items you'd like to add to your cart (e.g., 'Add item 1 and 3')\n"
        answer += "• Say 'add all to cart' to add all products\n"
        answer += "• Or ask me to find more specific products!\n\n"
        answer += "💡 **Pro Tip**: I can help you add items to your cart and complete your purchase seamlessly!"

        return {"answer": answer.strip(), "asins": asins, "products": products}

    except Exception as e:
        logger.error(f"Error in search_products: {e}")
        return {
            "answer": f"An error occurred while searching for products: {str(e)}. Please try rephrasing your search or try again later.",
            "asins": [],
            "products": [],
        }


def generate_smart_packing_list(question: str) -> List[str]:
    """
    Generate a smart packing list based on trip details using enhanced logic.
    
    Args:
        question: Trip description
        
    Returns:
        List of packing items
    """
    question_lower = question.lower()
    packing_items = []
    
    # Base essentials for any trip
    base_items = ["travel backpack", "phone charger", "toiletry bag", "travel adapter"]
    
    # Duration-based items
    if any(word in question_lower for word in ['week', '7 day', 'long trip']):
        base_items.extend(["laundry detergent pods", "extra underwear", "medication organizer"])
    elif any(word in question_lower for word in ['weekend', '2 day', '3 day', 'short trip']):
        base_items.extend(["travel size toiletries", "compact packing cubes"])
    
    # Climate and destination-based items
    climate_items = {
        'tropical': ["sunscreen SPF 50", "insect repellent", "lightweight clothing", "sandals", "sun hat"],
        'beach': ["swimsuit", "beach towel", "waterproof phone case", "flip flops", "beach bag"],
        'cold': ["thermal underwear", "winter jacket", "warm gloves", "beanie", "wool socks"],
        'mountain': ["hiking boots", "rain jacket", "first aid kit", "headlamp", "trekking poles"],
        'city': ["comfortable walking shoes", "day pack", "portable charger", "city guidebook"],
        'business': ["business attire", "laptop bag", "dress shoes", "iron travel size", "business cards"],
        'camping': ["sleeping bag", "camping tent", "camping stove", "water purification tablets", "multi-tool"]
    }
    
    # Match climate/activity keywords
    for climate, items in climate_items.items():
        if any(keyword in question_lower for keyword in [climate, climate.replace('_', ' ')]):
            packing_items.extend(items)
            break
    
    # Activity-specific items
    activities = {
        'photography': ["camera equipment", "extra batteries", "memory cards", "lens cleaning kit"],
        'fitness': ["workout clothes", "running shoes", "fitness tracker", "protein bars"],
        'cooking': ["portable cooking set", "spices travel kit", "cooler bag", "cutting board"],
        'reading': ["e-reader", "reading light", "book stand", "blue light glasses"]
    }
    
    for activity, items in activities.items():
        if activity in question_lower:
            packing_items.extend(items)
    
    # Combine and deduplicate
    all_items = base_items + packing_items
    return list(dict.fromkeys(all_items))  # Remove duplicates while preserving order


def generate_packing_list(user_id: str, question: str) -> Dict[str, Any]:
    """
    Process a user request to generate a packing list with product recommendations.
    Uses enhanced AI logic to generate a comprehensive packing list and then searches 
    Amazon for product recommendations for each item using SerpAPI.

    Args:
        user_id: The unique identifier of the user for whom products are being searched.
        question: User's query text requesting packing list (e.g., "I'm going to Hawaii for a week")

    Returns:
        Dict: called packing_list with results
            - 'answer': Formatted packing list with product recommendations
            - 'asins': Dict mapping packing list items to ASINs
            - 'items': List of packing list items with product details
    """
    try:
        logger.info(f"Generating enhanced packing list for user {user_id}: {question}")

        # Generate smart packing list
        packing_items = generate_smart_packing_list(question)
        
        # Limit items to avoid too many API calls (prioritize most important)
        priority_items = packing_items[:10]
        
        # Search for products for each packing item
        results = []
        asins_dict = {}
        failed_searches = []

        answer = f"🎒 **Smart Packing List for: {question}**\n\n"

        for item in priority_items:
            logger.info(f"Searching products for packing item: {item}")
            
            # Try multiple search variations for each item
            search_queries = [
                f"travel {item}",
                f"best {item}",
                item
            ]
            
            products = []
            for search_query in search_queries:
                search_results = search_amazon_products(search_query, max_results=4)
                if search_results["success"] and search_results["products"]:
                    products = search_results["products"][:3]  # Top 3 results
                    break
            
            if products:
                item_asins = [p["asin"] for p in products if p.get("asin")]
                asins_dict[item] = item_asins

                answer += f"📦 **{item.title()}**\n"
                answer += "   🏆 **Top Recommendations:**\n"

                for i, product in enumerate(products, 1):
                    # Format price
                    if isinstance(product["price"], (int, float)):
                        price_str = f"${product['price']:.2f}"
                    else:
                        price_str = str(product["price"])
                    
                    # Truncate long titles
                    title = product['title'][:50] + "..." if len(product['title']) > 50 else product['title']
                    
                    answer += f"   {i}. **{title}**\n"
                    answer += f"      💰 {price_str}"
                    
                    if product.get("rating") and product["rating"] > 0:
                        stars = "⭐" * int(product["rating"])
                        answer += f" | {stars} {product['rating']}/5"
                    
                    if product.get("prime"):
                        answer += " | 🚚 Prime"
                        
                    answer += f"\n      🛒 ASIN: {product['asin']}\n"

                answer += "\n"
                results.append({"item": item, "products": products})
            else:
                failed_searches.append(item)
                answer += f"📦 **{item.title()}**\n"
                answer += "   ❌ No specific products found - search manually\n\n"

        # Add summary
        if results:
            answer += f"✅ **Found products for {len(results)} items**\n"
            if failed_searches:
                answer += f"⚠️ **Manual search needed for:** {', '.join(failed_searches)}\n"
            answer += "\n💡 **Tip:** Ask me to add any items to your cart!"
        else:
            answer = f"Unable to find specific product recommendations for your packing list: {question}. Try searching for individual items."

        return {"answer": answer.strip(), "asins": asins_dict, "items": results}

    except Exception as e:
        logger.error(f"Error in generate_packing_list: {e}")
        return {
            "answer": f"An error occurred while generating your packing list: {str(e)}. Please try again with more specific trip details.",
            "asins": {},
            "items": [],
        }
