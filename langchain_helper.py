from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()
embeddings = OpenAIEmbeddings()


def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301")
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k})
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the {context}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose, detailed and no more then 5 sentences.
        """,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
    )
    
    response = qa_chain({"question": query})
    return response["answer"]

def generate_titles(db, query='Can you recommend me three better titles for this video?', k=3):
    query = "Can you recommend me three better titles for this video?"
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature = 0.8)
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be listed out, seperated with line character.

        Each title with 50 characters maximum.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)

    # titles = [title for title in response.split('-')[1::2]]
    # res = ""
    # for i, title in enumerate(titles):
    #     res = res + str(i+1) + ". " + title + "\n"

    return response

def summerize_text(db, k=4):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.8)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":k})

    prompt = PromptTemplate (
        input_variables=["context", "question"],
        template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
    
        Answer the following question: {question}
        By searching the {context}
    
        Only use the factual information from the transcript to answer the question.
    
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Use three sentences maximum. Keep the answer as concise as possible.
        """
    )

    qa_chain = RetrievalQA.from_chain_type(llm,
                                            retriever=retriever,
                                            chain_type_kwargs={"prompt": prompt})
    
    question = "Can you summarize the text for me?"
    response = qa_chain({"query": question})
    return response["result"]