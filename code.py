!pip install -q langchain-community llama-index sentence-transformers faiss-cpu pypdf chromadb transformers accelerate bitsandbytes jedi>=0.16

from google.colab import files
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

uploaded = files.upload()    #Upload document
if not uploaded:
    print("No files uploaded. Exiting.")
    exit()

loader = PyPDFLoader(list(uploaded.keys())[0])        #Load and split documents
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(texts, embeddings)    #Create vector store

model_name = "gpt2"    #Load Language Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

print("QA system is ready! Type 'quit' to exit.\n")
while True:
    question = input("Ask a question: ")
    if question.lower() in ['quit', 'exit', 'q']:
        break
    
    docs = db.similarity_search(question, k=2)        #Get relevancy from the document
    context = " ".join([d.page_content for d in docs])
    
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = llm(prompt)
    
    print("\nAnswer:", answer)
    print("\n" + "="*50 + "\n")

print("\nGoodbye!")
