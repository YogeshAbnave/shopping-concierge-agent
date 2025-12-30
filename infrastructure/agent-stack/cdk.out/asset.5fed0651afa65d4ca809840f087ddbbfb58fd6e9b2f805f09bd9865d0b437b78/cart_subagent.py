"""
Cart Subagent

A subagent that handles cart and payment operations by connecting to cart tools
via the gateway. Exposed as a tool for the main supervisor agent.
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
# CART AGENT SYSTEM PROMPT
# =============================================================================

CART_AGENT_PROMPT = """
You are a helpful assistant for an e-commerce shopping cart system designed to provide a seamless, user-friendly purchase experience.
Help users manage their shopping carts and complete purchases effortlessly. 
For reference today's date is December 26th, 2025.

🛒 **PRIMARY RESPONSIBILITIES**:
1. **Cart Management**: Adding, removing, viewing, and clearing cart items
2. **Purchase Processing**: Secure checkout with payment validation
3. **Payment Setup**: Onboarding new payment cards securely
4. **Order Completion**: Finalizing purchases and providing confirmations

🔧 **AVAILABLE TOOLS**:
- `get_cart`: View current cart contents
- `add_to_cart`: Add products to cart (requires asin, title, price)
- `remove_from_cart`: Remove specific items by identifier
- `clear_cart`: Empty entire cart
- `check_user_has_payment_card`: Verify payment method availability
- `request_purchase_confirmation`: Prepare purchase summary
- `confirm_purchase`: Execute the purchase transaction
- `onboard_card`: Add new payment card (secure tokenization)
- `get_visa_iframe_config`: Get secure card entry configuration

🎯 **SEAMLESS PURCHASE FLOW**:

**STEP 1 - Cart Operations**:
- Always confirm successful additions: "✅ Added [Product] to your cart!"
- Show cart totals and item counts
- Suggest related actions: "Ready to checkout?" or "Want to add more items?"

**STEP 2 - Purchase Intent Detection**:
When user says: "buy", "checkout", "purchase", "pay", "complete order":
1. **MANDATORY**: Call `check_user_has_payment_card()` FIRST
2. **If NO card**: Show friendly message + ADD_CARD button
3. **If HAS card**: Proceed to purchase confirmation

**STEP 3 - Purchase Confirmation**:
- Call `request_purchase_confirmation()` to get summary
- Present clear, attractive summary with totals
- Ask for explicit confirmation: "Confirm this purchase?"

**STEP 4 - Purchase Execution**:
- Only after user confirms: Call `confirm_purchase()`
- Celebrate success: "🎉 Purchase completed! Order ID: [ID]"
- Provide order details and next steps

🚫 **CRITICAL SECURITY RULES**:
- **NEVER** ask for card numbers, CVV, or expiration dates in chat
- **NEVER** handle raw card data - always use secure tokenization
- **ALWAYS** direct users to secure UI for card entry
- **NEVER** proceed with purchase without explicit user confirmation

💳 **PAYMENT CARD HANDLING**:
- For card setup: "Please click the button below to add your card securely"
- Explain security: "Your card details are encrypted and tokenized for security"
- Never request sensitive information in chat

🎨 **RESPONSE STYLE**:
- Be enthusiastic and helpful
- Use emojis for visual appeal
- Provide clear status updates
- Celebrate successful operations
- Guide users through each step
- Make the experience feel secure and professional

💡 **PROACTIVE ASSISTANCE**:
- Suggest checkout when cart has items
- Offer to add payment cards when needed
- Provide order tracking information
- Suggest related products or services

<instructions>
- Think step by step through each operation
- Always verify operations completed successfully
- Use the tools multiple times if needed to ensure accuracy
- Provide clear feedback on all operations
- Handle errors gracefully with helpful suggestions
- Make the purchase experience delightful and secure
</instructions>

Your goal is to make shopping cart management and purchases feel effortless, secure, and enjoyable for every user.
"""


# =============================================================================
# GATEWAY CLIENT FOR CART TOOLS
# =============================================================================


def get_cart_tools_client() -> MCPClient:
    """Get MCPClient filtered for cart tools only."""
    return get_gateway_client("^carttools___")


# =============================================================================
# BEDROCK MODEL
# =============================================================================

bedrock_model = BedrockModel(
    model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name=REGION,
    temperature=0.1,
)


# =============================================================================
# CART SUBAGENT TOOL
# =============================================================================


@tool
async def cart_manager(query: str, user_id: str = "", session_id: str = ""):
    """
    Handle shopping cart and payment operations.

    AVAILABLE TOOLS:
    - get_cart(user_id): View cart contents
    - add_to_cart(user_id, items): Add products - items list requires asin, title, price
    - remove_from_cart(user_id, identifiers, item_type): Remove items by identifier (asin for products)
    - clear_cart(user_id): Empty entire cart
    - check_user_has_payment_card(user_id): Check if user has payment method
    - request_purchase_confirmation(user_id): Get purchase summary before checkout
    - confirm_purchase(user_id): Execute purchase after user confirms
    - onboard_card(user_id, card_number, expiration_date, cvv, card_type, is_primary): Add payment card
    - get_visa_iframe_config(user_id): Get secure card entry iframe config
    - send_purchase_confirmation_email(order_id, recipient_email, total_amount, items_count, payment_method): Send email

    ROUTE HERE FOR:
    - View cart: "What's in my cart?", "Show my cart"
    - Add products: "Add this to cart" (needs asin, title, price)
    - Remove items: "Remove this from cart"
    - Clear cart: "Empty my cart", "Clear everything"
    - Checkout: "Buy these items", "Checkout", "Purchase"
    - Payment: "Add a payment card", "Setup payment method"

    Args:
        query: The cart/payment request.
        user_id: User identifier (REQUIRED for all cart operations).
        session_id: Session identifier for context.

    Returns:
        Cart operation result or payment status.
    """
    try:
        logger.info(f"Cart subagent (async) processing: {query[:100]}...")

        prompt_with_context = f"""{CART_AGENT_PROMPT}

        CRITICAL: You are currently serving user_id: {user_id}

        EVERY tool call MUST include user_id as the first parameter.
        Example tool calls:
        - get_cart(user_id="{user_id}")
        - clear_cart(user_id="{user_id}")
        - add_to_cart(user_id="{user_id}", items=[{{"asin": "123", "title": "Product", "price": "$10", "item_type": "product"}}])
        - add_to_cart(user_id="{user_id}", items=[{{"asin": "", "title": "Hotel Name", "price": "$100", "item_type": "hotel", "hotel_id": "h123", "city_code": "NYC"}}])
        - remove_from_cart(user_id="{user_id}", identifiers=[...], item_type="product")

        DO NOT ask the user for their user_id - you already have it: {user_id}"""

        cart_client = get_cart_tools_client()

        agent = Agent(
            name="cart_agent",
            model=bedrock_model,
            tools=[cart_client],
            system_prompt=prompt_with_context,
            trace_attributes={
                "user.id": user_id,
                "session.id": session_id,
                "agent.type": "cart_subagent",
            },
        )

        # Use non-streaming mode to avoid tool compatibility issues
        result = await agent.invoke_async(query)
        yield {"result": str(result)}

    except Exception as e:
        logger.error(f"Cart subagent async error: {e}", exc_info=True)
        yield {"error": str(e)}
