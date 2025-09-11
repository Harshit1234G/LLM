# ✨ Agentic AI Research Assistant
This project automates end-to-end research report generation. It searches multiple sources (Wikipedia, arXiv, news), extracts and organizes knowledge, writes structured report sections, critiques and refines them, and finally assembles everything into a professional PDF with references and appendices. The system is designed to be cost-efficient, reliable, and adaptable across topics, handling both well-documented and sparse domains.

## 🚀 Examples of Generated Reports
1. [Artificial Intelligence in Climate Modeling](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Artificial%20Intelligence%20in%20Climate%20Modeling.pdf)
2. [Attention Is All You Need](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Attention%20Is%20All%20You%20Need.pdf)
3. [Bruno Mars's Music Career](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Bruno%20Mars's%20Music%20Career.pdf)
4. [CRISPR Gene Editing Technology](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/CRISPR%20Gene%20Editing%20Technology.pdf)
5. [Calculus](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Calculus.pdf)
6. [Cryovolcanoes on Enceladus](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Cryovolcanoes%20on%20Enceladus.pdf)
7. [Harry Potter](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Harry%20Potter.pdf)
8. [Impact of Social Media on Mental Health](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Impact%20of%20Social%20Media%20on%20Mental%20Health.pdf)
9. [Quantum Computing](https://github.com/Harshit1234G/LLM/blob/main/Researcher/results/Quantum%20Computing.pdf)

## 🌟 Features
* Automated end-to-end research pipeline with modular agents.
* Sources from **Wikipedia**, **arXiv**, and **recent news**.
* Self-critique and revision loop for higher-quality reports.
* Structured PDF output with references and appendices.
* Automatic **Vancouver-style citations**.
* Progress saved in JSON to prevent data loss.
* Execution time and logs for transparency.
* Simple, interactive CLI for ease of use.

## 🛠️ Tech Stack
| Category           | Technologies                     |
| ------------------ | -------------------------------- |
| **Languages**      | Python                           |
| **APIs**           | Wikipedia, arXiv, GNews          |
| **LLM & AI Tools** | OpenAI API, LangChain, LangGraph |
| **Others**         | Git, venv, markdown-pdf          |

## 🔄 Workflow
> Fore more details check the docstrings in the code.
<img width="600" height="600" alt="researcher" src="https://github.com/user-attachments/assets/7dc7b39b-da55-45e9-a748-00e77ca24734" />

## ⚙️ Installation
Step-by-step instructions:
```bash
git clone https://github.com/Harshit1234G/LLM
cd Researcher
pip install -r requirements.txt
```

## 🚀 Usage
1. **Get API Keys**
   * Create an [OpenAI API Key](https://platform.openai.com/account/api-keys) to enable LLM calls.
   * Create a [LangSmith API Key](https://smith.langchain.com/) to enable tracing and project logging.
   
2. **Set Environment Variables**
   You can either:
   * Create a `.env` file in the project root with the following content:
    ```env
    OPENAI_API_KEY=your_openai_key_here
    LANGSMITH_API_KEY=your_langsmith_key_here
    LANGSMITH_TRACING=true
    LANGSMITH_ENDPOINT=https://api.smith.langchain.com
    LANGSMITH_PROJECT=your_langsmith_project_name_here
    ```
   * Or, enter these values every time you run the program directly (you’ll be prompted to paste these values when missing).

3. **Run the Research Assistant**
    ```bash
    python main.py
    ```

## 📁 Project Structure
```
Researcher/
├── agents/                     # contains all the agents
│   └── __init__.py
│   └── assembler.py
│   └── base_agent.py
│   └── critic.py
│   └── extractor.py
│   └── orchestration.py        # creates the main pipeline
│   └── searcher.py
│   └── writer.py
├── config/
│   └── __init__.py
│   └── settings.py             # some basic configuration settings
├── data/                       # stores the intermediate data as json files
├── logs/                       # ignored, but it will store the logs
├── results/                    # final generated reports
├── tools/
│   └── __init__.py
│   └── api_wrappers.py         # wrappers for Wikipedia & arXiv APIs
│   └── news.py                 # wrapper for gnews API
├── utils/
│   └── __init__.py
│   └── caching.py              # contains saving & loading functions
│   └── logger.py               # logger
│   └── methodology.txt
├── main.py
├── README.md
└── requirements.txt
```

## 🧩 Challenges & Learnings
Building this project involved overcoming several challenges, such as designing an efficient multi-agent workflow, handling inconsistent data availability (e.g., niche topics with limited sources), and ensuring cost-effective LLM usage. Through this, I learned how to orchestrate agentic AI systems, manage state across multiple agents, and integrate tools like LangGraph, LangSmith, and markdown-to-PDF pipelines. This project greatly improved my skills in building reliable, production-style AI applications.

## 🔮 Future Improvements
- Adding images to the final report.
- Removing the current limitations:
    * A max of 3 Wikipedia articles & 3 arXiv research papers can be retrieved.
    * Retrieve complete documents, not just upto 12,000 characters.

## 📜 License
This project is licensed under the [Apache License](https://github.com/Harshit1234G/LLM/blob/main/LICENSE).

## 🙏 Acknowledgements
- [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph) for LLM orchestration.
- [OpenAI](https://openai.com/) for powering the LLM.
- [arXiv](https://arxiv.org/) for research papers.
- [Wikipedia](https://www.wikipedia.org/) for articles.
- [GNews](https://pypi.org/project/gnews/) for recent news.
- [markdown-pdf](https://pypi.org/project/markdown-pdf/) for converting the markdown to pdf.