import os
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
from flask import Flask, request
import logging


load_dotenv()

logging.basicConfig(level=logging.ERROR)


def fetch_job_description(url):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the script tag with the specific attribute
        script_tag = soup.find("script", {"data-automation": "server-state"})

        # Extract and process the content of the script tag
        if script_tag and script_tag.string:
            script_content = script_tag.string

            # Find the line containing `window.SEEK_APOLLO_DATA`
            for line in script_content.splitlines():
                if "window.SEEK_APOLLO_DATA" in line:
                    return line.split("window.SEEK_APOLLO_DATA = ")[1]
        else:
            print("No <script> tag with data-automation='server-state' or content found.")
            return None
    else:
        print(f"Failed to fetch the URL. Status code: {response.status_code}")
        return None


def generate_coverletter(resume, job_description):
    model_local = ChatOpenAI(model_name="gpt-4")

    # Define text splitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )

    # Split and store resume
    resume_splits = text_splitter.split_text(resume)
    resume_vectorstore = Chroma.from_texts(
        texts=resume_splits,
        collection_name="resume-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory=None
    )
    resume_retriever = resume_vectorstore.as_retriever()

    # Split and store job description
    job_splits = text_splitter.split_text(job_description)
    job_vectorstore = Chroma.from_texts(
        texts=job_splits,
        collection_name="job-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory=None
    )
    job_retriever = job_vectorstore.as_retriever()

    # Cover letter prompt
    after_rag_template = """
    Cover Letter Creator focuses on generating the main content of the cover letter without any headings or personal contact information.
    It directly crafts a narrative that highlights the user's skills and experiences relevant to the job description, maintaining a formal tone throughout.
    The GPT is designed to start the cover letter with an introduction relevant to the job role and company,
    and if any additional information is needed for accuracy, it will request it from the user.
    Do not hallucinate or make up things that aren't in my Resume.
    Resume: {resume}
    Job Description: {job_description}

    When you reieve the Trigger "Go" you will generate the cover letter
    Trigger: {trigger}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = ({"resume": resume_retriever, "job_description": job_retriever, "trigger": RunnablePassthrough()} | after_rag_prompt | model_local | StrOutputParser())

    cover_letter = after_rag_chain.invoke("Go")
    return cover_letter


app = Flask(__name__)


def check_auth():
    """Check if Authorization header is valid"""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return False, "Missing Authorization header"

    # Expecting format: "Bearer <token>"
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False, "Invalid Authorization header format"

    token = parts[1]
    if token != os.getenv("MASTER_KEY"):
        return False, "Invalid token"

    return True, "Authorized"


@app.route('/namejeff', methods=['POST'])
def namejeff():
    is_authorized, message = check_auth()
    if not is_authorized:
        return message, 401

    try:
        jeff = request.form['jeff']

        model_local = ChatOpenAI(model_name="gpt-4")
        after_rag_template = "Just say 'Jeff'"
        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
            {"job_description": RunnablePassthrough()} | after_rag_prompt | model_local | StrOutputParser()
        )
        cover_letter = after_rag_chain.invoke(jeff)
        return cover_letter, 200
    except Exception as e:
        logging.error(f"Error occured {e}")
        return "Something went wrong", 500


@app.route('/submit', methods=['POST'])
def submit():
    is_authorized, message = check_auth()
    if not is_authorized:
        return message, 401
    try:
        if 'resume' not in request.form or 'url' not in request.form:
            return 'Missing resume or url', 400

        resume = request.form['resume']
        seek_url = request.form['url']

        job_description = fetch_job_description(seek_url)

        if job_description is None:
            return 'Could not fetch job decription from Seek', 500

        cover_letter = generate_coverletter(resume, job_description)
        return cover_letter, 200
    except Exception as e:
        logging.error(f"Error occured {e}")
        return "Something went wrong", 500


@app.route('/ping', methods=['GET'])
def ping():
    is_authorized, message = check_auth()
    if not is_authorized:
        return message, 401
    return "", 200


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8000)
