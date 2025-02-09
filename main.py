from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
from docx import Document
import argparse

load_dotenv()


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
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_text(resume)

    vectorstore = Chroma.from_texts(
        texts=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """
    Cover Letter Creator focuses on generating the main content of the cover letter without any headings or personal contact information.
    It directly crafts a narrative that highlights the user's skills and experiences relevant to the job description, maintaining a formal tone throughout.
    The GPT is designed to start the cover letter with an introduction relevant to the job role and company,
    and if any additional information is needed for accuracy, it will request it from the user.
    Do not hallucinate or make up things that aren't in my Resume
    Resume: {resume}
    Job Description: {job_description}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"resume": retriever, "job_description": RunnablePassthrough()} | after_rag_prompt | model_local | StrOutputParser()
    )
    cover_letter = after_rag_chain.invoke(job_description)
    return cover_letter


def text_to_pdf(text, filename="cover_letter.docx"):
    doc = Document()

    for line in text.split("\n"):
        doc.add_paragraph(line)

    doc.save(filename)
    print(f"DOCX saved as {filename}")


def main():
    parser = argparse.ArgumentParser(description="AI Cover Letter Generator! Pass in a Markdown file containing your resume and a link to the Job Description you want on Seek to generate a cover letter")
    parser.add_argument("-f", "--file", type=str, help="Path to your Resume in Markdown format.")
    parser.add_argument("-u", "--url", type=str, help="URL to the Job Posting on Seek i.e. https://www.seek.co.nz/job/81759577")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            resume = f.read()
    else:
        print("Error: Please specify a Markdown file containing your resume with the -f or --file flag")
        exit(1)

    if args.url:
        seek_url = args.url
    else:
        print("Error: Please specify a url to a Seek job posting with the -u or --url flag i.e. https://www.seek.co.nz/job/81759577")
        exit(1)

    print("Fetching Job Description...")
    job_description = fetch_job_description(seek_url)
    print("done...")

    if job_description is not None:
        print("Generating cover letter...")
        cover_letter = generate_coverletter(resume, job_description)
        print("Finished creating cover letter...\n\n")
        print(cover_letter)


if __name__ == "__main__":
    main()
