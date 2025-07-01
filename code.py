!pip install -q langchain-community llama-index sentence-transformers faiss-cpu pypdf chromadb transformers accelerate bitsandbytes jedi>=0.16

from google.colab import files
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# Upload document
uploaded = files.upload()
if not uploaded:
    print("No files uploaded. Exiting.")
    exit()

# Load and split documents
loader = PyPDFLoader(list(uploaded.keys())[0])
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
db = FAISS.from_documents(texts, embeddings)

# Load LLM
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Interactive Q&A loop
print("QA system is ready! Type 'quit' to exit.\n")
while True:
    question = input("Ask a question: ")
    if question.lower() in ['quit', 'exit', 'q']:
        break
    
    # Get relevant documents
    docs = db.similarity_search(question, k=2)
    context = " ".join([d.page_content for d in docs])
    
    # Generate answer
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = llm(prompt)
    
    print("\nAnswer:", answer)
    print("\n" + "="*50 + "\n")

print("\nGoodbye!")
