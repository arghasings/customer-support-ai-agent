from langchain_openai import ChatOpenAI

def get_sentiment(text):
    text = text.lower()

    # Angry / Negative
    if any(word in text for word in [
        "angry", "upset", "frustrated", "not delivered",
        "late", "delay", "worst", "bad", "issue", "problem",
        "deducted", "failed", "error"
    ]):
        return "Angry 😡"

    # Happy / Positive
    elif any(word in text for word in [
        "happy", "good", "great", "thanks", "thank you",
        "awesome", "perfect"
    ]):
        return "Happy 😊"

    # Neutral
    else:
        return "Neutral 😐"