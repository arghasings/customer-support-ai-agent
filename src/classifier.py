import os
from langchain_openai import ChatOpenAI

def classify_ticket(query):
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt = f"""
        Classify the following query into:
        Delivery, Billing, Technical, General

        Query: {query}
        """

        return llm.invoke(prompt).content.strip()

    # fallback
    if "delay" in query.lower():
        return "Delivery"
    elif "refund" in query.lower():
        return "Billing"
    elif "error" in query.lower() or "not working" in query.lower():
        return "Technical"
    else:
        return "General"