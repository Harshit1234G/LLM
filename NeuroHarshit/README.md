# ✨ NeuroHarshit
An AI-powered personal portfolio assistant built using RAG (Retrieval-Augmented Generation). The system is designed to answer queries about my background, projects, skills, certifications, education, and experience.

### 🔗 Live Demo
[NeuroHarshit App](https://harshit1234g-llm-neuroharshitmain-cvzhgc.streamlit.app/) 🚀

## 🚀 Preview
<img width="1439" height="811" alt="Preview of Streamlit UI" src="https://github.com/user-attachments/assets/0097ac37-5f76-40e0-af31-97c907426f81" />

## 🌟 Features
- Answers questions about me.
- Uses RAG with FAISS for efficient retrieval.
- Displays complete chat history.
- You can ask follow-up questions.
- Three deployments:
    - CLI for quick local testing
    - Streamlit app for interactive usage
    - API endpoint for integration into other apps

## 🛠️ Tech Stack
| Category           | Technologies                     |
| ------------------ | -------------------------------- |
| **Languages**      | Python                           |
| **Frameworks**     | Streamlit                        |
| **LLM & AI Tools** | OpenAI API, LangChain, LangGraph |
| **Database**       | FAISS                            |
| **Others**         | Git, venv                        |

## 🔄 Workflow
<img width="600" height="600" alt="workflow" src="https://github.com/user-attachments/assets/43b82b3f-c74c-48d7-a917-b11257aa2673" />

## ⚙️ Installation
- Step-by-step instructions:
```bash
git clone https://github.com/Harshit1234G/NeuroHarshit.git
cd NeuroHarshit
pip install -r requirements.txt
streamlit run main.py
```

## 📚 Knowledge Base
- The chatbot’s knowledge comes from plain text files stored in the `Databases/text_data` folder.  
- Currently, it includes 8 files:  
    - `basic.txt`  
    - `certifications.txt`  
    - `education.txt`  
    - `experience.txt`  
    - `faq.txt`  
    - `hobbies.txt`  
    - `projects_summary.txt`  
    - `skills.txt`  
- You can fully customize the knowledge base:  
    - Replace the existing files with your own text files.  
    - File names do **not** need to match the existing ones.  
    - You can add, remove, or combine files — the system will automatically index whatever text files are provided.
    - If you change the text data, you have to run `Agent/vector_db.ipynb` file for creating the FAISS index.

## 🚀 Usage
- **CLI Interface** → Run the `Agent/chatbot.py` directly.
- **API** → You can customize or use the API (`api.py`) as per your needs.
- **Streamlit App** → Launch the web app:
```bash
streamlit run main.py
```
- **API Key Setup**
    * For **CLI & API**: Either keep an `.env` file with `OPENAI_API_KEY` or provide the key when prompted during runtime.
    * For **Streamlit**: Enter your API key in the sidebar.
    * Get your API key here → [OpenAI API Keys](https://platform.openai.com/account/api-keys)

* **Customization** → Easily adapt the chatbot to yourself. For example, just change "Harshit" to your name in prompts.

## 📁 Project Structure
- A small tree view of important files/folders:
```
NeuroHarshit/
├── Agent/
│   └── chatbot_graph.png
│   └── chatbot.py
│   └── testing.ipynb
│   └── vector_db.ipynb
├── Databases/
│   └── faiss_index/
│   └── text_data/
├── api.py
├── main.py
├── README.md
└── requirements.txt
```

## ⚡ Challenges & Learnings  
- Building a reliable RAG system for consistent answers.  
- Handling conversation state across different deployments (CLI, API, Streamlit).  
- Learned best practices for structuring modular AI projects.  

## 🔮 Future Improvements  
- Proper hosting of the website.
- Integrating with my Github and LinkedIn, for a dynamic experience, code explanations and better visualizations.
- Enhance UI/UX of the Streamlit app.

## 📜 License  
This project is licensed under the [Apache License](https://github.com/Harshit1234G/LLM/blob/main/LICENSE).  

## 🙏 Acknowledgements  
- [Streamlit](https://streamlit.io/) for the interactive web app framework.  
- [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph) for LLM orchestration.  
- [FAISS](https://faiss.ai/) for vector search.  
- [OpenAI](https://openai.com/) for powering the LLM.  