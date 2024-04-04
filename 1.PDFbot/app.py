import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def getTextFromPDFs(pdfDocs):
    text = ""
    for pdf in pdfDocs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def getChunksFromText(rawText):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(rawText)
    return chunks

def getEmbeddingsFromChunks(text_chunks):
    # Paid Embeddings( GPT 3.5 or 4 )
    # https://openai.com/pricing ( ADA V2)
    # embeddings = OpenAIEmbeddings()

    # Unpaid Embeddings
    # instructor-xl Is Embedding Models Which Is Used To Get Embeddings From The Chunks. It Is Slower If We Are Using It On Computer But It Is Too Fast If We Have Our Own Hardware
    # https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    # For Embeddings It Is Better Then Any Other Large Language Model.
    # It Ranks Higher Then OpenAi Embeddings.
    # To User Instructor Embeddings  , We Have To Install More Dependencies.
    # pip install InstructorEmbedding sentence_transformers
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return embeddings

def storeEmbeddingsIntoVectorStore(textChunks , chunksEmbeddings):
    vectorStore = FAISS.from_texts(texts=textChunks , embedding=chunksEmbeddings)
    return vectorStore

def getConversationChain(vectorStore):
    '''
    Langchain Allow Us To Add Memory In Question System.
    It Means We Ask Question And Then Ask Follow Up Question About Previous Answers.
    Langchain Provide Bydefaul Memory System To Chatbot.
    '''
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # Initialize The Memory Instances.
    memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True)

    # Initialize The Chat Session(Question And FollowUp Question).
    # This Conversation Chain Allow Use To Chat With Our Vectore Store.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handlingUserInput(userQuestion):
    response = st.session_state.conversation({'question': userQuestion})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    # Load The Dotenv File.
    load_dotenv()

    # Set The Page Configuration( Chrome Tab )
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # It Sets The Header Of Page And TextInput
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handlingUserInput(user_question)

    # Add CSS At The Top of The File.
    st.write(css, unsafe_allow_html=True)

    # It Sets The Content Inside Sidebar.
    with st.sidebar:
        st.subheader("Your documents")
        pdfDocs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        # For Multiple PDFs = ( accept_multiple_files = True )
        if st.button("Process"):
            with st.spinner("Processing"):


                # Extract Text From PDFs.
                rawText = getTextFromPDFs(pdfDocs)

                # Divide The Text Into Chunks.
                textChunks = getChunksFromText(rawText)

                # Create Embeddings From Text Chunks.
                chunksEmbeddings = getEmbeddingsFromChunks(textChunks)

                # Store All These Embeddings Into Vectore Store.
                vectorStore  = storeEmbeddingsIntoVectorStore(textChunks , chunksEmbeddings)

                # Create Conversation Chain.

                '''
                This Conversation Chain Allow Use To Chat With Our Vectore Store.
                It Will Allow Us To Generate New Message Of Conversation.
                It Will Take History Of The Conversation And Returns The Next Element In Conversation.

                conversation = getConversationChain(vectorStore)

                When We Click The Button OR Submits The Form , Streamlit Reloads Entire Code Section.So It Is Going To Reinitializes All
                The Variables.If We Want To Save Some Variable After OnClik Or OnSubmit , We Have To use 'st.session_state'

                st.session_state.conversation = getConversationChain(vectorStore)

                After Using st.session_state , We Are Going To State That The Variable Is Not Going To Reinitialize After Reloading It.

                We Can Also Use This convesation In Outside Of The Function.

                This Is The Way To Make The Variables Persistent.
                
                '''
                
                st.session_state.conversation = getConversationChain(vectorStore)



if __name__ == "__main__":
    main()