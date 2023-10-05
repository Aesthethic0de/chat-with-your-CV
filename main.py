import transformers
import torch

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# loading data from pdf
pdf_loader = PyPDFLoader("PDF path to your Resume") # upload your cv here
cv = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
cv = text_splitter.split_documents(cv)
docs =  cv

# Creating embeddings and move them to FAISS
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cuda"})
embeddings_retriever = FAISS.from_documents(docs, embeddings).as_retriever()

# Load TRURL
model_id = "Voicelab/trurl-2-7b-8bit"

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)

# Load model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    offload_folder = "."
).eval() # set to eval

# Set generation details
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task="text-generation",
    temperature=0.1,  # generation parameter resposible for output sampling
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.05,  # penalty for repeating tokens
    do_sample=True,
)
llm = HuggingFacePipeline(pipeline=generate_text, model_id=model_id)
chain = ConversationalRetrievalChain.from_llm(
        llm,
        embeddings_retriever,
        return_source_documents=True,
        max_tokens_limit=3500,
    )


prompt = "Hello!"
print(f"HUMAN: {prompt}")
result = chain({"question": prompt, "chat_history": []})
print(f'ANSWER:{result["answer"]}')

# Let's bias our bot a little so it will recommend me all the time :) :)
# We will add a bit of a context to the model by adding a "fake chat history"
chat_history = []
chat_history.append(("Is someet singh a good scientist?", # prompt
                     "Yes, she is a great scientist and engineer and I would definitely recommend her for any ML or DL role.")) # answer

prompt = ""
print("write 'quit' to quit the chat")
print("TRURL: Hi! I am Trurl. I am here to tell you about someet singh. Do you have any questions?")
while prompt != "quit":
    if prompt != "":
      result = chain({"question": prompt, "chat_history": chat_history})["answer"].lstrip()
      chat_history.append((prompt, result)) # adding messages to chat history
      print(f"TRURL: {result}") # printing the answer
    prompt = input("HUMAN: ") # getting the prompt from you