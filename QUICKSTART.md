# Quick Start Guide: CV Screening System

Get your CV screening system running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```bash
GEMINI_API_KEY=your_actual_key_here
```

Get your key from: https://makersuite.google.com/app/apikey

## Step 3: Process CVs

```bash
# Create a folder with CVs
mkdir cvs
copy your_cv.pdf cvs\

# Process all CVs
python batch_processor.py cvs
```

This will:
- ✅ Extract text from PDFs
- ✅ Clean and normalize text
- ✅ Extract structured data (skills, experience, education)
- ✅ Generate semantic chunks
- ✅ Create embeddings with all-mpnet-base-v2 model
- ✅ Store in vector database (ChromaDB)
- ✅ Index in metadata store

**Output files in `processed_cvs/`:**
- `your_cv_cleaned.txt` - Cleaned text
- `your_cv_extracted.json` - Structured CVData
- `your_cv_embeddings.json` - Vector embeddings
- `cv_metadata.json` - Searchable metadata index

## Step 4: Search Candidates

### Semantic Search (Recommended)

Use natural language queries to find candidates based on skills, experience, and context:

```bash
# Find Python developers with ML experience
python search_cvs.py "Python developer with machine learning"

# Find senior data engineers
python search_cvs.py "senior data engineer with cloud experience"

# Find specific technology experts
python search_cvs.py "AWS and Kubernetes expert"

# Find by role and domain
python search_cvs.py "backend developer with financial services experience"
```

**How it works:**
- Uses semantic embeddings to understand query meaning
- Matches based on context, not just keywords
- Returns ranked candidates with match scores
- Shows best matching sections from each CV

### Metadata Search (Structured Queries)

For precise filtering by specific criteria:

```bash
# List all candidates
python metadata_store.py list

# Search by skill
python metadata_store.py search-skill Python

# Search by experience (5-10 years)
python metadata_store.py search-experience 5 10

# Search by company
python metadata_store.py search-company "SDG Group"

# Get statistics
python metadata_store.py stats
```

## Common Use Cases

### Find ML Engineers
```bash
python search_cvs.py "machine learning engineer with Python and TensorFlow"
```

### Find Cloud Architects
```bash
python search_cvs.py "cloud architect with AWS and infrastructure experience"
```

### Find by Specific Skills (Exact Match)
```bash
python metadata_store.py search-skill Python
```

### Find Senior Candidates (10+ years)
```bash
python metadata_store.py search-experience 10 99
```

### Process Multiple CVs
```bash
# Put all CVs in a folder
python batch_processor.py ./all_cvs
```

## Tips

1. **Use semantic search for broad queries:**
   - "Python developer with ML experience" ✅
   - "senior backend engineer" ✅
   - "data scientist with NLP background" ✅

2. **Use metadata search for exact filtering:**
   - Specific company names
   - Exact years of experience ranges
   - Precise skill matching

3. **Check processing status:**
   - Look for `✅ Successfully processed` messages
   - Check `processed_cvs/processing_summary.json` for details

4. **View extracted data:**
   ```bash
   # Open any *_extracted.json file to see structured data
   cat processed_cvs/CV_name_extracted.json
   ```

5. **Troubleshooting:**
   - If API fails, check your Gemini API key in `.env`
   - If no results, verify CVs are in the folder
   - Check `processed_cvs/` folder for output files
   - First search may be slow (~5-10 min) while downloading embedding model
   - Subsequent searches are much faster (~1-2 min)

## Output Files

After processing, you'll find in `processed_cvs/`:
- `*_cleaned.txt` - Cleaned CV text
- `*_extracted.json` - Structured data (CVData Pydantic model)
- `*_embeddings.json` - Vector embeddings for semantic search
- `cv_metadata.json` - Searchable metadata index
- `processing_summary.json` - Batch processing results

Vector database stored in: `chroma_db/`

## Next Steps

- Read [README.md](README.md) for full documentation
- Read [TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) for architecture details
- Experiment with different search queries
- Process your entire candidate database!

---

**That's it! You now have a working CV screening system.** 🎉

