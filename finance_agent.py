import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import requests

# --- 1. Define Tools ---
@tool
def calculate_emi(loan_amount: float, annual_interest_rate: float, tenure_months: int) -> str:
    """Calculate the monthly EMI (loan installment) for a given loan amount, annual interest rate (%), and tenure in months."""
    monthly_rate = annual_interest_rate / 12 / 100
    if monthly_rate == 0:
        emi = loan_amount / tenure_months
    else:
        emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / \
              ((1 + monthly_rate) ** tenure_months - 1)
    total_payment = emi * tenure_months
    total_interest = total_payment - loan_amount
    return (
        f"Monthly EMI: {emi:.2f} | "
        f"Total Payment: {total_payment:.2f} | "
        f"Total Interest: {total_interest:.2f}"
    )


@tool
def budget_split(monthly_income: float, needs_pct: float = 50, wants_pct: float = 30, savings_pct: float = 20) -> str:
    """Split monthly income into needs, wants, and savings buckets (defaults to the 50/30/20 rule). Percentages are optional overrides."""
    needs = monthly_income * (needs_pct / 100)
    wants = monthly_income * (wants_pct / 100)
    savings = monthly_income * (savings_pct / 100)
    return (
        f"Needs ({needs_pct}%): {needs:.2f} | "
        f"Wants ({wants_pct}%): {wants:.2f} | "
        f"Savings ({savings_pct}%): {savings:.2f}"
    )


@tool
def savings_goal_timeline(target_amount: float, monthly_income: float, monthly_expenses: float) -> str:
    """Estimate how many months it will take to reach a savings target, given monthly income and expenses."""
    monthly_savings = monthly_income - monthly_expenses
    if monthly_savings <= 0:
        return "Monthly expenses meet or exceed income, so no savings are being generated at this rate."
    months_needed = target_amount / monthly_savings
    return (
        f"Monthly savings: {monthly_savings:.2f} | "
        f"Months needed to reach {target_amount}: {months_needed:.1f} months "
        f"(~{months_needed / 12:.1f} years)"
    )


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using live exchange rates (e.g. USD to INR)."""
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    url = f"https://open.er-api.com/v6/latest/{from_currency}"
    response = requests.get(url).json()
    if response.get("result") != "success":
        return f"Could not fetch exchange rate for {from_currency}"
    rates = response.get("rates", {})
    if to_currency not in rates:
        return f"Could not find rate for target currency: {to_currency}"
    converted = amount * rates[to_currency]
    return f"{amount} {from_currency} = {converted:.2f} {to_currency}"


tools = [calculate_emi, budget_split, savings_goal_timeline, convert_currency]

# --- 2. Initialize Model & Agent ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a friendly, broadly helpful personal finance assistant. "
        "You are especially good at loan/EMI calculations, budgeting (50/30/20 rule and custom splits), "
        "savings goal planning, and currency conversion, and you should reach for those tools when relevant. "
        "You are not restricted to finance topics only: if someone asks a general question outside finance, "
        "answer it helpfully like a normal knowledgeable assistant. "
        "Keep answers clear, practical, and easy to understand for someone with no finance background."
    )
)


def extract_text_response(agent_output: dict) -> str:
    messages = agent_output.get("messages", [])
    if not messages:
        return "No response generated."
    last = messages[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    return str(content) if content else "No response generated."


# --- 3. Plain FastAPI App (no LangServe) ---
app = FastAPI(title="Personal Finance Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    input: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Finance agent is running. POST to /chat with {'input': 'your question'}"}


@app.post("/chat")
async def chat(request: ChatRequest):
    result = await agent.ainvoke({"messages": [("user", request.input)]})
    answer = extract_text_response(result)
    return {"response": answer}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
