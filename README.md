Johor State Election 2026 Chatbot

A chatbot that answers your questions about the 2026 Johor state election using news articles from The Star.

Step 1: Install the project's tools

pip install -r requirements.txt


Step 2: Download the AI models

Install the Ollama to runs the AI on your computer: https://ollama.com/download

Make sure Ollama is open. It runs in the background after you install it. Then run these two commands one at a time:

ollama pull mxbai-embed-large
ollama pull llama3.2:3b

The download is about 3 GB so it can take a while.

Step 3: Collect the news articles and build the database

python scrape.py

This script:
1. collects the Johor election articles from The Star and saves them to johor_election_articles.json and .csv,
2. splits them into small chunks and turns them into AI vectors,
3. saves everything into a local database folder called chroma_db.

The scraping will take a while because it waits a little between pages to be polite to the website.

After scrape finish you will see:

all done! now can run:  streamlit run app.py


Step 4: Start the chatbot

streamlit run app.py

Type a question at the bottom of the page. For example:
- Who won the Johor state election?
- When is the polling day for the Johor election?

To stop the chatbot go back to the terminal and press Ctrl + C.

_____________________________________________________

Using it again next time

You don't need to repeat Steps 1 to 3. Just make sure Ollama is open then run:

streamlit run app.py
