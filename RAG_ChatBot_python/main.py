import torch
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import CHROMA_DB_PATH, EMBEDDING_MODEL, SEARCH_K
from api import get_llm

print("Запуск RAG бота...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

print("Загрузка базы...")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DB_PATH),
    embedding_function=embeddings
)

llm = get_llm()

retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
prompt = PromptTemplate.from_template(
    """
    Ты - ассистент, отвечающий на вопросы по книге "Python. Исчерпывающее руководство".
    Используй ТОЛЬКО информацию из контекста ниже для ответа.
    Если ответа нет в контексте, честно скажи: "Я не нашел ответа в книге".
    Отвечай на русском языке, подробно и понятно.

    Контекст:\n{context}\n\n
    Вопрос: {question}\n\n
    Ответ:
    """
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def chat(question, history):
    try:
        result = qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

demo = gr.ChatInterface(
    fn=chat,
    title="🐍 Python RAG Bot",
    description="Задавайте вопросы по книге Python",
    examples=["Что такое декораторы?", "Как работают списки?", "Объясни классы"]
)

if __name__ == "__main__":
    print("✅ Бот готов! Откройте ссылку в браузере...")
    demo.launch()