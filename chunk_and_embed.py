import os
import json
import requests
import chromadb

# use free local api only
ollama_url = "http://localhost:11434/api/embed"
embed_model = "mxbai-embed-large"

# find files inside this project folder (works even if run from another folder)
this_folder = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(this_folder, "johor_election_articles.json")

# use chromadb as local database
db_folder = os.path.join(this_folder, "chroma_db")
collection_name = "johor_election"

# mxbai-embed-large window limit is 512 tokens which roughly around 300
# so keep each chunk around 250 words to leave space for articel title
max_words = 250

# Count how many words are in the text
def count_words(text):
    return len(text.split())

# Split a long paragraph into smaller pieces
def split_long_paragraph(paragraph):
    # if 1 paragraph is too long just cut it by words
    words = paragraph.split() # split paragraph into individual words
    pieces = []

     # take 250 words at one time
    for i in range(0, len(words), max_words):
        pieces.append(" ".join(words[i:i + max_words]))
    return pieces

# Split the full article into smaller chunks
def make_chunks(content):
    paragraphs = []

    # Read the article paragraph by paragraph
    for p in content.split("\n"):
        p = p.strip() # remove unnecessary spaces
        if p == "":
            continue  # skip empty paragraphs

        # split paragraphs that are longer than 250 words
        if count_words(p) > max_words:
            paragraphs.extend(split_long_paragraph(p))
        else:
            paragraphs.append(p)

    chunks = []
    current = []


    # combine paragraphs until the chunk reaches 250 words
    for p in paragraphs:
        current_words = count_words(" ".join(current))

        # Check whether adding the next paragraph exceeds 250 words
        if current_words + count_words(p) > max_words and len(current) > 0:
            chunks.append("\n".join(current))

            # do overlap where kee the last paragraph for the next chunk
            last_paragraph = current[-1]
            if count_words(last_paragraph) + count_words(p) <= max_words:
                current = [last_paragraph]
            else:
                current = []

        current.append(p)

    # save the remaining text as the final chunk
    if len(current) > 0:
        chunks.append("\n".join(current))

    return chunks


# EMBEDDING USING OLLAMA MODEL

def get_embeddings(text_list):
    data = {
        "model": embed_model,
        "input": text_list
    }
    response = requests.post(ollama_url, json=data)
    result = response.json()
    return result["embeddings"]



# RUN EVERYTHING (chunk + embed + save into chromadb)
# so that scrape.py can call function easily

def build_database(articles):

    # prepare empty lists to store info chunks
    all_ids = []
    all_texts = []
    all_metadatas = []


    # go through each article
    for article in articles:

        # split article content into smaller chunks
        chunks = make_chunks(article["content"])

        # and then go through each chunks
        for i, chunk in enumerate(chunks):
            # add title and date to every chunk
            text = "Title: " + article["title"] + "\n"
            text = text + "Date: " + article["publish_date"] + "\n\n"
            text = text + chunk

            # declare each chunk a uniqueID and store all info about the article
            all_ids.append(article["url"] + "#chunk" + str(i))
            all_texts.append(text)
            all_metadatas.append({
                "title": article["title"],
                "publish_date": article["publish_date"],
                "author": article["author"],
                "section": article["section"],
                "url": article["url"],
                "chunk_no": i,
                "total_chunks": len(chunks)
            })


    ids = []
    texts = []
    metadatas = []
    for i in range(len(all_ids)):
        ids.append(all_ids[i])
        texts.append(all_texts[i])
        metadatas.append(all_metadatas[i])

    if len(ids) == 0:
        return

    # convert text chunks into numerical embeddings
    embeddings = []
    batch_size = 32

    # process is 32 chunks at a time
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # see embedding progress
        print("embedding", i + 1, "to", i + len(batch), "of", len(texts))
        # generate embeddings for this batch
        embeddings.extend(get_embeddings(batch))

    # create the local chromadb folder
    client = chromadb.PersistentClient(path=db_folder)

    # delete old collection if got any so we dont get duplicate data when run again
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # first time run nothing to delete

    # create a new collection that use cosine distance for similarity search
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # save 100 at a time, dont save in one go
    for i in range(0, len(ids), 100):
        collection.add(
            ids=ids[i:i + 100],
            documents=texts[i:i + 100],
            embeddings=embeddings[i:i + 100],
            metadatas=metadatas[i:i + 100]
        )

    print("saved", collection.count(), "chunks into", db_folder)
    print("all done! now can run:  streamlit run app.py")


# run locally just in case
if __name__ == "__main__":
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    build_database(articles)
