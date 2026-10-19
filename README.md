# AiThatDontLie 

**A Retrieval-Augmented Generation (RAG) system that knows when to answer, search again, or abstain.**

AiThatDontLie explores how to make AI systems more reliable by grounding their answers in retrieved evidence and allowing them to say *"I don't know"* when the available information does not support a confident answer.

The project compares a standard RAG pipeline with an uncertainty-aware alternative, investigating how evidence quality and confidence signals can reduce unsupported answers, even when a user pressures the model to agree with a particular point of view.

## 🎯 Project Goals

- **Evidence-grounded answers:** Use retrieved documents to support generated responses.
- **Uncertainty awareness:** Recognise when the available evidence is insufficient to answer a question reliably.
- **Abstention:** Allow the model to decline to answer rather than inventing unsupported information.
- **Retry and retrieval:** Search for additional evidence when the initial retrieval results are insufficient.
- **Comparative evaluation:** Compare a basic RAG baseline against an abstention-enabled pipeline.
- **Transparency:** Expose retrieved sources, similarity scores, citations, and decision signals to help users understand the system's output.

## 🧠 How It Works

AiThatDontLie uses Retrieval-Augmented Generation (RAG) to combine information retrieval with language-model generation.

The system retrieves relevant passages from a reference corpus and uses them to inform its response. The abstention-aware pipeline adds checks intended to determine whether the retrieved evidence is sufficient to support an answer.

The system can make three decisions:

| Decision | Meaning |
|---|---|
| 🟢 `ANSWER` | The available evidence supports generating an answer. |
| 🟡 `SEARCH_AGAIN` | The initial evidence is insufficient, so the system attempts a broader retrieval before answering. |
| 🔴 `ABSTAIN` | The evidence does not sufficiently support an answer, so the system declines to provide one. |

The application also displays the generated answer, relevant decision explanations, and retrieved passages with their similarity scores and citation information.

## 🏗️ Architecture

The project brings together several components:

1. **Document corpus:** A collection of reference articles used as the knowledge base.
2. **Text embeddings:** Documents and queries are represented as vectors to support semantic retrieval.
3. **Vector search:** FAISS is used to find passages relevant to a user's question.
4. **Language model:** An LLM generates answers grounded in the retrieved context and supports evidence-related checks.
5. **Uncertainty layer:** Retrieval relevance and answer-support signals help determine whether to answer, retry retrieval, or abstain.
6. **Interactive interface:** A Streamlit application lets users compare the baseline and abstention-aware pipelines.

The current demonstration uses a tennis-focused corpus of approximately 24 Wikipedia articles.

## ✨ Features

- Interactive question-answering interface built with Streamlit.
- Side-by-side selection between **Basic RAG (baseline)** and **Abstaining RAG**.
- Semantic document retrieval using sentence embeddings.
- FAISS-based similarity search.
- Configurable retrieval and uncertainty thresholds.
- Additional retrieval when the initial evidence is insufficient.
- Optional self-consistency checks.
- Visibility into retrieved sources, similarity scores, and citations.
- A configurable architecture intended for experimentation and evaluation.

## 🛠️ Technology Stack

- **Python** — application logic and RAG pipeline.
- **Streamlit** — interactive web interface.
- **Sentence Transformers** — text embeddings using `all-MiniLM-L6-v2`.
- **FAISS** — vector similarity search.
- **Anthropic Claude** — language-model integration, configured for `claude-haiku-4-5`.

See `requirements.txt` and the project configuration for the implementation details and dependencies.

## 🚀 Getting Started

### Prerequisites

- Python installed on your system.
- Git.
- An Anthropic API key if required by the configured language-model integration.
- A terminal or command prompt.

### 1. Clone the repository

```bash
git clone https://github.com/PCTUD/AiThatDontLie.git
cd AiThatDontLie
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

If the project requires API credentials, use the provided `.env.example` as a reference for the expected variables.

Copy it to `.env`:

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Populate the required values in `.env`. Never commit API keys or other secrets to version control.

### 5. Launch the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to interact with the application.

> **Note:** If the application reports missing corpus files, an index, or environment variables, check the repository's ingestion workflow and configuration before running it. The exact initial data-preparation steps depend on the current implementation.

## 💬 Usage

1. Launch the Streamlit application.
2. Choose a pipeline:
   - **Abstaining RAG** — can abstain when the evidence does not support a reliable answer.
   - **Basic RAG (baseline)** — provides a comparison against a conventional pipeline.
3. Enter a question about tennis or another topic covered by the available corpus.
4. Review the generated answer and its decision status.
5. Expand **Sources & signals** to inspect the retrieved passages, similarity scores, and citations.

Try asking a question that is well covered by the corpus, followed by one for which the corpus contains little or no relevant information. Compare how the two pipelines respond.

## ⚙️ Configuration

The project's `config.py` centralises important retrieval, model, and uncertainty settings.

| Parameter | Purpose |
|---|---|
| `EMBED_MODEL` | Sentence embedding model used for semantic retrieval. |
| `CHUNK_CHARS` | Approximate text chunk size. |
| `CHUNK_OVERLAP` | Overlap between adjacent chunks. |
| `TOP_K` | Number of passages retrieved during the initial search. |
| `TOP_K_RETRY` | Number of passages retrieved during a broader search. |
| `TAU_RETRIEVE` | Minimum retrieval similarity threshold. |
| `TAU_CONSISTENCY` | Similarity threshold for the optional self-consistency check. |
| `TREAT_PARTIAL_AS_UNSUPPORTED` | Controls how partially supported answers are treated. |
| `ENABLE_SELF_CONSISTENCY` | Enables or disables the additional consistency signal. |
| `LLM_MODEL` | Configured language model. |
| `LLM_MAX_TOKENS` | Maximum generated token count. |
| `LLM_TEMPERATURE` | Generation temperature. |

These values influence the trade-off between answering more questions and avoiding answers that lack sufficient support. Adjust thresholds carefully and evaluate changes against representative questions.

## 🔬 Research Direction

The central question behind AiThatDontLie is:

**Can a RAG system be made more resistant to unsupported claims and user pressure by explicitly modelling uncertainty?**

Potential areas of investigation include:

- Comparing answer accuracy between the baseline and abstention-aware pipelines.
- Measuring how frequently the system abstains on questions with insufficient evidence.
- Evaluating whether retrieval similarity correlates with answer correctness.
- Investigating the trade-off between answering coverage and reliability.
- Testing how the system behaves when prompts contain leading questions or pressure to agree.
- Exploring whether answer consistency across different evidence subsets improves reliability.

A key evaluation challenge is balancing **coverage** (how often the system answers) against **reliability** (how often its answers are adequately supported). Excessive abstention makes a system less useful, while insufficient abstention can lead to confident but unsupported claims.

## ⚠️ Limitations

AiThatDontLie is an experimental project, not a guarantee of factual correctness.

- Retrieved documents may be incomplete, outdated, or incorrect.
- Similarity scores measure semantic relevance, not factual truth.
- A language model can still produce unsupported or misleading statements.
- Abstention thresholds require evaluation and may need adjustment for different datasets.
- Performance on a small tennis-focused corpus does not establish performance across other domains.
- User pressure resistance must be measured experimentally rather than assumed.

The goal is to investigate and improve evidence-grounded behaviour, not to claim that hallucinations can be eliminated entirely.

## 📁 Project Structure

```text
AiThatDontLie/
├── data/              # Corpus, index, and evaluation data
├── src/               # RAG and retrieval implementation
├── .env.example       # Example environment configuration
├── .gitignore
├── app.py             # Streamlit interface
├── config.py          # Model, retrieval, and uncertainty settings
├── requirements.txt   # Python dependencies
└── README.md
```

Additional generated files and evaluation utilities may be present or added as the project develops.

## 🤝 Contributing

Contributions, suggestions, and experiments are welcome.

1. Fork the repository.
2. Create a branch for your change.
3. Make your changes and test them locally.
4. Submit a pull request explaining the change and its motivation.

Ideas for contributions include improving evaluation methodology, expanding the reference corpus, refining abstention criteria, and testing resistance to leading or adversarial prompts.

For bugs or feature requests, open an issue in the [GitHub repository](https://github.com/PCTUD/AiThatDontLie/issues).

## 📄 License

Please check the repository for its current licence before redistributing or using this project in other work.

## 👤 Author

Maintained by [PCTUD](https://github.com/PCTUD).

---

*Built to explore a simple principle: a reliable AI system should know when its evidence is not good enough to answer.*
