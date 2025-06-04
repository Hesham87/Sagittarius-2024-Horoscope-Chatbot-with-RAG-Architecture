# Sagittarius-2024-Horoscope-Chatbot-with-RAG-Architecture

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Building a chatbot that offers horoscope predictions for individuals born under the Sagittarius sign in 2024. By focusing on a specific subject area, the chatbot can provide more accurate and relevant responses compared to generic chatbots.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         sagittarius_2024_horoscope_chatbot_with_rag_architecture and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── sagittarius_2024_horoscope_chatbot_with_rag_architecture   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sagittarius_2024_horoscope_chatbot_with_rag_architecture a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Sagittarius: A RAG-Based Horoscope Chatbot

Sagittarius is a Retrieval-Augmented Generation (RAG) chatbot designed to answer astrology-related questions using a curated horoscope dataset. By integrating LangChain, Pinecone, HuggingFace embeddings, and a local LLaMA model, it delivers concise, context-aware responses grounded in the provided data.

---

## 🔹 Features

- **RAG Architecture**  
  Combines document retrieval and generation for informed answers.

- **Semantic Search**  
  Uses HuggingFace embeddings + Pinecone to retrieve relevant horoscope snippets.

- **Re-Ranking**  
  Employs a CrossEncoder (MS-Marco MiniLM L6-v2) to prioritize the most pertinent chunks.

- **Local LLaMA Model**  
  Generates answers based solely on the retrieved context (no external API calls).
  **Llama 2-7B** for response generation (quantized version)

- **Astrology Focus**  
  Tailored to respond to user questions about horoscopes, star signs, and related topics.

---

## 📥 Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/sagittarius.git
   cd sagittarius

