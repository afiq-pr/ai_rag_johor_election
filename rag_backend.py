import os
import requests
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# using local model. choose ollama because maybe powerful LLM for free version
ollama_host = "http://localhost:11434"
embed_model = "mxbai-embed-large"
chat_model = "llama3.2:3b"

# chromadb save the database as a normal folder inside this project
this_folder = os.path.dirname(os.path.abspath(__file__))
db_folder = os.path.join(this_folder, "chroma_db")
collection_name = "johor_election"

# how many relevant chunks to give to the LLM
top_k = 5

# if cosine distance is 0 means same meaning, 1 means not related at all
# if even the most relevant chunks is far than this number, the question is out of scope
max_distance = 0.5

# message when we cannot answer (out of scope)
sorry_message = "Sorry, I couldn't find relevant information in the available articles."

# Catchphrases that indicate that LLM cannot answer the question.
no_answer_phrases = [
    "none of the snippets",
    "no mention",
    "not mentioned",
    "no information",
    "do not contain",
    "does not contain",
    "do not provide",
    "does not provide",
    "cannot answer",
    "can't answer"
]



# message when the database is not created yet
no_database_message = "The article database is not ready yet. Please run  python scrape.py  first, then restart the chatbot."

# Connect to the local chromadb folder
client = chromadb.PersistentClient(path=db_folder)
try:
    client.get_collection(collection_name)
    database_ready = True
except Exception:
    # the collection only exists after running scrape.py (or chunk_and_embed.py)
    database_ready = False

# langchain part: only used to search the closest chunks
# OllamaEmbeddings turn the question into numbers using the same local embed model
embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_host)

# connect langchain to the same chromadb collection we built in chunk_and_embed.py
vector_store = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embeddings
)


def search_articles(question):
    # embed model uses this instruction to know we want to search for relevant passages
    search_text = "Represent this sentence for searching relevant passages: " + question

    # find the closes chunks
    # it returns a list of document and score. score is cosine distance (lower = more similar)
    results = vector_store.similarity_search_with_score(search_text, k=top_k)

    # Go through every matching news snippet.
    snippets = []
    for doc, distance in results:
        meta = doc.metadata

        # Keep only the information we need for answering and showing sources
        snippets.append({
            "text": doc.page_content,
            "title": meta["title"],
            "date": meta["publish_date"][:10],
            "author": meta["author"],
            "section": meta["section"],
            "url": meta["url"],
            "distance": distance
        })
    return snippets



# Generate the answer
def make_prompt(question, snippets):
    # put a number on each snippet so the LLM can cite like [1], [2]
    context = ""
    for i, snip in enumerate(snippets):
        context = context + "[" + str(i + 1) + "] "
        context = context + "Title: " + snip["title"] + " | Date: " + snip["date"] + "\n"
        context = context + snip["text"] + "\n\n"

    prompt = "Here are some news snippets from The Star about the Johor state election:\n\n"
    prompt = prompt + context
    prompt = prompt + "Question: " + question + "\n\n"
    prompt = prompt + "Answer the question using only the snippets above, in one short paragraph. "
    prompt = prompt + "Do not repeat the same fact. "
    prompt = prompt + "After each fact, put the snippet number you used, like [1] or [2]. "
    prompt = prompt + "If NONE of the snippets can answer the question, reply with only the word NO_ANSWER."
    return prompt


system_message = (
    "You are a helpful assistant that answers questions about the Johor state election "
    "using only news snippets from The Star. Never make up facts. "
    "If the answer is not in the snippets, reply with only the word NO_ANSWER."
)


def ask_llm(prompt):
    # Prepare the information that we want to send to ollama
    data = {
        "model": chat_model,
        # 1. system message is for general rules for the LLM.
        # 2. user message is for news snippets and the actual question.
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "stream": False, # wait for the complete answer instead of show word by word
        "options": {"temperature": 0}  # 0 means less creative, hence less hallucinate
    }
    # send the prompt to ollamao chat API
    response = requests.post(ollama_host + "/api/chat", json=data)
    result = response.json()
    return result["message"]["content"].strip()


# Put everything together
# Create this function for front end
# receive question and return
def answer_question(question):

    # check whether the user submitted an empty question
    if question.strip() == "":
        return {"answer": "Please type a question.", "sources": []}

    # check whether the database is created already
    if not database_ready:
        return {"answer": no_database_message, "sources": []}

    # search chromadb for relevant snippets
    snippets = search_articles(question)
    # if chromadb returns nothing then show the fallback message
    if len(snippets) == 0:
        return {"answer": sorry_message, "sources": []}

    # check whether the closest snippet is relevant enough
    # First result is the closest match because chromadb sorts by distance
    best_distance = snippets[0]["distance"]

    # if the closest snippet is far from the question then,,
    # we treat the question as out of scope
    if best_distance > max_distance:
        return {"answer": sorry_message, "sources": []}

    # only keep snippets that are close enough
    good_snippets = []
    for s in snippets:
        if s["distance"] <= max_distance:
            good_snippets.append(s)

    # prepare the question and snippets for the LLM and ask llm to generate answer
    prompt = make_prompt(question, good_snippets)
    answer = ask_llm(prompt)

    # check whether can answer or not
    # if it return NO_ANSWER show fallback message
    if answer.startswith("NO_ANSWER"):
        return {"answer": sorry_message, "sources": []}

    # small model sometimes still write NO_ANSWER at the end of an answer, so remove it
    answer = answer.replace("NO_ANSWER", "").strip()

    # it could not find an answer in the snippet
    for phrase in no_answer_phrases:
        if phrase in answer.lower():
            return {"answer": sorry_message, "sources": []}

    # only show the sources the LLM really cited
    sources = []
    seen_titles = []
    for i, s in enumerate(good_snippets):
        tag = "[" + str(i + 1) + "]"
        if tag in answer and s["title"] not in seen_titles:
            seen_titles.append(s["title"])
            sources.append({
                "number": i + 1,
                "title": s["title"],
                "date": s["date"],
                "url": s["url"]
            })

    # if the LLM forgot to cite just show all the snippets we gave it
    if len(sources) == 0:
        for i, s in enumerate(good_snippets):
            if s["title"] not in seen_titles:
                seen_titles.append(s["title"])
                sources.append({
                    "number": i + 1,
                    "title": s["title"],
                    "date": s["date"],
                    "url": s["url"]
                })

    return {"answer": answer, "sources": sources}

# quick test in local terminal
if __name__ == "__main__":
    test_questions = [
        "Who won the Johor state election?",
        "When is the polling day for the Johor election?",
        "What is the best recipe for nasi lemak?"  # out of scope should say sorry
    ]

    for q in test_questions:
        print("\nQuestion:", q)
        result = answer_question(q)
        print("Answer:", result["answer"])
        for s in result["sources"]:
            print("  [" + str(s["number"]) + "]", s["title"], "(" + s["date"] + ")")
