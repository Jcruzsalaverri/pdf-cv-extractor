# CV Screening System - README

A specialized Python system for processing, analyzing, and screening CVs using LLM-powered extraction and RAG (Retrieval Augmented Generation).

## 📢 Release Notes (2025-11-26)

### **Single-Chunk Architecture & API Rotation**

- **Single Comprehensive Chunk**: Each CV is now processed as a single, comprehensive text chunk containing all structured data (skills, experience, education). This improves search accuracy by providing full context to the embedding model.
- **API Key Rotation**: Implemented automatic rotation for Gemini API keys to eliminate rate limit issues during batch processing.
- **Local Embeddings**: Switched to local `all-mpnet-base-v2` model exclusively for embeddings (faster, free, private).
- **Simplified Codebase**: Consolidated chunking logic and removed unused modules.

---

## 🎯 What This System Does

**Transform CVs → Searchable Database → Intelligent Screening**

1. **Extract & Clean** - Pull text from PDFs and normalize formatting
2. **Structure Data** - Extract skills, experience, education using LLMs
3. **Enable Search** - Semantic + metadata search across candidates
4. **Rank & Compare** - Find the best candidates for your role

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEYS (comma-separated)
```

**Example `.env`:**
```bash
# Comma-separated list for rotation
GEMINI_API_KEYS=key1,key2,key3,key4
```

### 3. Process CVs

```bash
# Create folder with CVs
mkdir cvs
copy *.pdf cvs\

# Process all CVs
python batch_processor.py cvs
```

### 4. Search Candidates

```bash
# Semantic search
python search_cvs.py "senior data engineer with cloud experience"

# Metadata search
python metadata_store.py search-skill Python
```

---

## 📋 Core Pipeline

### **`batch_processor.py`** - Main Orchestrator

Processes CVs through the complete pipeline:

```bash
python batch_processor.py ./cvs_folder
```

**What it does:**

1. ✅ Extract text from PDFs (`extract_text.py`)
2. ✅ Clean formatting artifacts (`text_cleaner.py`)
3. ✅ Extract structured data (`cv_extractor.py`)
4. ✅ Format as single comprehensive chunk (`chunking.py`)
5. ✅ Generate embeddings locally (`embedding.py`)
6. ✅ Store in vector database (`vector_store.py`)
7. ✅ Index in metadata store (`metadata_store.py`)

**Output (in `data/` folder):**

- `CV_name_cleaned.txt` - Cleaned text
- `CV_name_extracted.json` - Structured data
- `CV_name_embeddings.json` - Embeddings
- `cv_metadata.json` - Searchable index

---

## 🔍 Search & Query

### Metadata Search (Exact Matches)

```bash
# Search by skill
python metadata_store.py search-skill "Python"

# Search by experience
python metadata_store.py search-experience 5 10  # 5-10 years

# Search by company
python metadata_store.py search-company "SDG Group"

# List all candidates
python metadata_store.py list

# Get statistics
python metadata_store.py stats
```

### Individual Components

```bash
# Extract text only
python extract_text.py CV.pdf output.txt

# Clean text only
python text_cleaner.py CV.txt

# Extract structured data only
python cv_extractor.py CV_cleaned.txt
```

---

## 📊 Extracted Data Structure

Each CV is transformed into structured JSON:

```json
{
  "candidate_name": "John Doe",
  "email": "john.doe@example.com",
  "phone": "+1 555 0123 456",
  "total_years_experience": 5.5,
  "current_role": "Senior Developer",
  "companies": ["Tech Corp", "StartUp Inc"],
  "roles": ["Senior Developer", "Junior Developer"],
  "technical_skills": ["Python", "Machine Learning", "Cloud Computing"],
  "programming_languages": ["Python", "JavaScript"],
  "frameworks": ["Django", "React"],
  "tools": ["Docker", "Git", "AWS"],
  "degrees": [
    {
      "degree": "Master",
      "field": "Computer Science",
      "university": "Tech University",
      "year": "2018-2020"
    }
  ],
  "certifications": ["AWS Certified Solutions Architect"]
}
```

---

## 🏗️ Project Structure

```
pdf-agent/
├── Core Pipeline
│   ├── extract_text.py          # PDF → text
│   ├── text_cleaner.py          # Clean formatting
│   ├── cv_extractor.py          # Extract structured data
│   ├── chunking.py              # Format CV as single chunk
│   ├── embedding.py             # Generate local embeddings
│   └── batch_processor.py       # Orchestrate all steps
│
├── Database
│   ├── vector_store.py          # ChromaDB (semantic search)
│   └── metadata_store.py        # JSON index (structured search)
│
├── Configuration
│   ├── config.py                # Settings
│   ├── .env                     # API keys (gitignored)
│   ├── .env.example             # Template
│   └── requirements.txt         # Dependencies
│
├── Data (gitignored)
│   ├── data/                    # Processed CVs
│   └── chroma_db/              # Vector database
│
└── Documentation
    ├── README.md               # This file
    ├── QUICKSTART.md          # Quick start
    └── docs/
        ├── TECHNICAL_GUIDE.md
        ├── RAG_PIPELINE_EXPLAINED.md
        └── THEORETICAL_FOUNDATIONS.md
```

---

## ⚙️ Configuration

Edit `.env` file:

```bash
# LLM Provider (for text cleaning & extraction)
LLM_PROVIDER=gemini
GEMINI_API_KEYS=key1,key2,key3  # Comma-separated list

# Embedding Model (Local Only)
LOCAL_EMBEDDING_MODEL=all-mpnet-base-v2

# Retrieval
RETRIEVAL_TOP_K=5
```

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[SYSTEM_DOCUMENTATION.md](docs/TECHNICAL_GUIDE.md)** - Complete technical reference
- **[THEORETICAL_FOUNDATIONS.md](docs/THEORETICAL_FOUNDATIONS.md)** - Why RAG works
---

## 🔧 Requirements

- Python 3.8+
- PyMuPDF (PDF extraction)
- LangChain (LLM integration)
- ChromaDB (vector database)
- Google Gemini API key(s) (for LLM extraction)
- SentenceTransformers (for local embeddings)

Install all:

```bash
pip install -r requirements.txt
```

---

## 💡 Key Features

✅ **LLM-powered extraction** - Intelligent data extraction from any CV format  
✅ **Single-Chunk Context** - Full CV context in one vector for better matching  
✅ **API Key Rotation** - Automatically cycles keys to avoid rate limits  
✅ **Local Embeddings** - Free, fast, and private embedding generation  
✅ **Hybrid search** - Semantic (vector) + exact (metadata) search  
✅ **Batch processing** - Process hundreds of CVs automatically  

---

## 🎯 Use Cases

- **Recruitment agencies** - Screen large volumes of candidates
- **HR departments** - Find qualified candidates quickly
- **Talent acquisition** - Build searchable candidate databases
- **Career services** - Analyze and compare CVs

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

---

**Built with ❤️ using LangChain, ChromaDB, and Google Gemini**
