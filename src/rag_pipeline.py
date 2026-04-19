from langchain_openai import ChatOpenAI

def build_rag_chain(vectorstore):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )

    def qa_chain(query):
        docs = vectorstore.similarity_search(query, k=1)
        context = docs[0].page_content

        prompt = f"""
        Use this context to answer:

        Context: {context}
        Question: {query}
        """

        return llm.invoke(prompt).content

    return qa_chain