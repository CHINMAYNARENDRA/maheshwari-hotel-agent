import os
import random
import re
from typing import TypedDict, Literal
from difflib import SequenceMatcher
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────
#  MENU
# ─────────────────────────────────────────────
MENU = {
    "dal fry":               {"price": 170, "quantity": "1 bowl"},
    "dal tadka":             {"price": 180, "quantity": "1 bowl"},
    "dal makhani":           {"price": 250, "quantity": "1 bowl"},
    "kaju masala":           {"price": 330, "quantity": "1 bowl"},
    "kaju paneer masala":    {"price": 330, "quantity": "1 bowl"},
    "mushroom matar masala": {"price": 290, "quantity": "1 bowl"},
    "prawns sukka":          {"price": 470, "quantity": "8 pcs"},
    "prawns fry":            {"price": 470, "quantity": "8 pcs"},
    "prawns tandoori tikka": {"price": 470, "quantity": "8 pcs"},
    "fish tandoori tikka":   {"price": 500, "quantity": "8 pcs"},
}

ALIASES = {"daal": "dal", "dall": "dal", "prawn": "prawns", "prwans": "prawns"}

MENU_TEXT = "\n".join(
    f"  {i+1}. {name.title():<25} Rs.{info['price']}  ({info['quantity']})"
    for i, (name, info) in enumerate(MENU.items())
)

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
class State(TypedDict):
    user_query:      str
    llm_response:    str
    is_menu_related: bool
    dish:            str
    price:           int
    quantity:        str
    quantity_num:    int
    order_status:    str
    cook_status:     str
    serve_status:    str
    order_retries:   int
    cook_retries:    int
    failure_stage:   str
    steps:           list

MAX_RETRIES = 3

# ─────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

SYSTEM_PROMPT = f"""You are a warm and professional restaurant assistant named Raj for Maheshwari Hotel.

Our menu:
{MENU_TEXT}

RULES:
1. If customer greets or asks about menu → show the menu and ask what they'd like.
2. If customer names a dish → confirm: dish name, price, quantity. Keep it warm and concise.
3. If query is unrelated to food/restaurant → politely decline.
4. NEVER invent dishes, prices, or quantities not listed above.
5. End EVERY reply with exactly: [MENU_RELATED: YES] or [MENU_RELATED: NO]
"""

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def extract_quantity(text: str) -> int:
    """Extract quantity from text like '2 bowls', 'three plates', etc."""
    t = text.lower()
    
    # Try to find numbers
    numbers = re.findall(r'\b(\d+)\b', t)
    if numbers:
        return int(numbers[0])
    
    # Try word numbers
    word_numbers = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    for word, num in word_numbers.items():
        if word in t:
            return num
    
    return 1

def fuzzy_match(text: str, dish_name: str, threshold: float = 0.75) -> bool:
    """Fuzzy matching for typo tolerance."""
    ratio = SequenceMatcher(None, text.lower(), dish_name.lower()).ratio()
    return ratio >= threshold

def extract_dish(text: str):
    t = text.lower().strip()
    
    # Apply aliases
    for wrong, right in ALIASES.items():
        t = t.replace(wrong, right)
    
    # Extract quantity
    quantity_num = extract_quantity(t)
    
    # Exact match first
    for name, info in MENU.items():
        if name in t:
            return name.title(), info["price"], info["quantity"], quantity_num
    
    # Two-word match
    words = t.split()
    for name, info in MENU.items():
        for i in range(len(words) - 1):
            if " ".join(words[i:i+2]) in name:
                return name.title(), info["price"], info["quantity"], quantity_num
    
    # Fuzzy match for typos (only for longer words to avoid false positives)
    for name, info in MENU.items():
        for word in words:
            if len(word) > 3 and fuzzy_match(word, name.split()[0], threshold=0.75):
                return name.title(), info["price"], info["quantity"], quantity_num
    
    return "", 0, "", 1

# ─────────────────────────────────────────────
#  NODES
# ─────────────────────────────────────────────
def llm_node(state: State) -> State:
    query = state["user_query"]
    stage = state.get("failure_stage", "")
    if stage == "cook":
        query += "\n\n(Note: kitchen issue. Apologise briefly, confirm retrying.)"
    elif stage == "order":
        query += "\n\n(Note: order system error. Apologise and ask customer to confirm dish.)"

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    response = llm.invoke(messages)
    text = response.content

    is_menu = "[MENU_RELATED: YES]" in text
    clean = text.replace("[MENU_RELATED: YES]", "").replace("[MENU_RELATED: NO]", "").strip()

    dish, price, qty, qty_num = extract_dish(state["user_query"])

    steps = state.get("steps", [])
    steps.append({"stage": "llm", "status": "done", "message": clean})

    return {
        **state,
        "llm_response":    clean,
        "is_menu_related": is_menu,
        "dish":    dish  or state.get("dish", ""),
        "price":   price or state.get("price", 0),
        "quantity": qty  or state.get("quantity", ""),
        "quantity_num": qty_num or state.get("quantity_num", 1),
        "failure_stage": "",
        "steps": steps,
    }

def take_order(state: State) -> State:
    steps = state.get("steps", [])

    if not state.get("dish"):
        retries = state.get("order_retries", 0) + 1
        steps.append({"stage": "order", "status": "error", "message": f"No dish selected (attempt {retries}/{MAX_RETRIES})"})
        return {**state, "order_status": "failed", "order_retries": retries,
                "failure_stage": "order", "steps": steps}

    if random.random() < 0.2:
        retries = state.get("order_retries", 0) + 1
        steps.append({"stage": "order", "status": "error", "message": f"Order system error (attempt {retries}/{MAX_RETRIES})"})
        return {**state, "order_status": "failed", "order_retries": retries,
                "failure_stage": "order", "steps": steps}

    qty_num = state.get("quantity_num", 1)
    total_price = state["price"] * qty_num
    qty_display = f"{qty_num} bowls" if "bowl" in state["quantity"] else f"{qty_num}x {state['quantity']}"
    
    steps.append({"stage": "order", "status": "done",
                  "message": f"Order confirmed: {qty_num}x {state['dish']} @ Rs.{total_price} ({qty_display})"})
    return {**state, "order_status": "confirmed", "order_retries": 0,
            "failure_stage": "", "steps": steps}

def cook(state: State) -> State:
    steps = state.get("steps", [])

    if random.random() < 0.15:
        retries = state.get("cook_retries", 0) + 1
        steps.append({"stage": "cook", "status": "error", "message": f"Kitchen issue (attempt {retries}/{MAX_RETRIES})"})
        return {**state, "cook_status": "failed", "cook_retries": retries,
                "failure_stage": "cook", "steps": steps}

    steps.append({"stage": "cook", "status": "done", "message": f"{state['quantity_num']}x {state['dish']} cooked perfectly!"})
    return {**state, "cook_status": "cooked", "cook_retries": 0,
            "failure_stage": "", "steps": steps}

def serve(state: State) -> State:
    steps = state.get("steps", [])

    if random.random() < 0.1:
        steps.append({"stage": "serve", "status": "error", "message": "Serving hiccup — retrying from kitchen"})
        return {**state, "serve_status": "failed", "steps": steps}

    steps.append({"stage": "serve", "status": "done", "message": f"{state['quantity_num']}x {state['dish']} served!"})
    return {**state, "serve_status": "served", "steps": steps}

# ─────────────────────────────────────────────
#  EDGES
# ─────────────────────────────────────────────
def route_after_llm(state: State) -> Literal["take_order", "__end__"]:
    return "take_order" if state["is_menu_related"] else "__end__"

def route_after_order(state: State) -> Literal["llm_node", "cook", "__end__"]:
    if state["order_status"] == "confirmed": return "cook"
    if state.get("order_retries", 0) >= MAX_RETRIES: return "__end__"
    return "llm_node"

def route_after_cook(state: State) -> Literal["llm_node", "serve", "__end__"]:
    if state["cook_status"] == "cooked": return "serve"
    if state.get("cook_retries", 0) >= MAX_RETRIES: return "__end__"
    return "llm_node"

def route_after_serve(state: State) -> Literal["cook", "__end__"]:
    return "__end__" if state["serve_status"] == "served" else "cook"

# ─────────────────────────────────────────────
#  BUILD GRAPH
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(State)
    graph.add_node("llm_node",   llm_node)
    graph.add_node("take_order", take_order)
    graph.add_node("cook",       cook)
    graph.add_node("serve",      serve)
    graph.set_entry_point("llm_node")
    graph.add_conditional_edges("llm_node",   route_after_llm,
        {"take_order": "take_order", "__end__": END})
    graph.add_conditional_edges("take_order", route_after_order,
        {"llm_node": "llm_node", "cook": "cook", "__end__": END})
    graph.add_conditional_edges("cook",       route_after_cook,
        {"llm_node": "llm_node", "serve": "serve", "__end__": END})
    graph.add_conditional_edges("serve",      route_after_serve,
        {"cook": "cook", "__end__": END})
    return graph.compile()

# ─────────────────────────────────────────────
#  FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(title="Maheshwari Hotel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrderRequest(BaseModel):
    message: str

@app.post("/api/order")
async def place_order(req: OrderRequest):
    graph = build_graph()

    initial: State = {
        "user_query":      req.message,
        "llm_response":    "",
        "is_menu_related": False,
        "dish":            "",
        "price":           0,
        "quantity":        "",
        "quantity_num":    1,
        "order_status":    "pending",
        "cook_status":     "pending",
        "serve_status":    "pending",
        "order_retries":   0,
        "cook_retries":    0,
        "failure_stage":   "",
        "steps":           [],
    }

    final = graph.invoke(initial)

    total_price = final["price"] * final.get("quantity_num", 1)

    return {
        "llm_response":    final["llm_response"],
        "is_menu_related": final["is_menu_related"],
        "dish":            final["dish"],
        "price":           total_price,
        "quantity":        final["quantity"],
        "quantity_num":    final.get("quantity_num", 1),
        "order_status":    final["order_status"],
        "cook_status":     final["cook_status"],
        "serve_status":    final["serve_status"],
        "steps":           final["steps"],
    }

@app.get("/")
async def root():
    return FileResponse("frontend/maheshwari.html")

if __name__ == "__main__":
    import uvicorn
    print("\n🍛 Maheshwari Hotel server starting...")
    print("   Open http://localhost:8000 in your browser\n")
    uvicorn.run("restuarantagent:app", host="0.0.0.0", port=8000, reload=True)