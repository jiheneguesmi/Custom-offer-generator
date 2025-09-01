# Offer Generator

An intelligent Flask application that generates comprehensive offers using AI-powered document retrieval, natural language processing, and automated quality assessment.

##  Features

- **AI-Powered Offer Generation**: Automatically creates detailed proposals in French
- **Document Processing Pipeline**: Extracts and indexes content from PowerPoint presentations
- **Semantic Search**: Uses FAISS vectorstore with BGE embeddings for relevant document retrieval
- **Quality Control**: Multi-stage grading system for document relevance, hallucination detection, and answer quality
- **Intelligent Routing**: Automatically routes queries to appropriate data sources (vectorstore vs web search)
- **Comprehensive Metrics**: Tracks performance, quality, cost, and confidence scores
- **LangSmith Integration**: Full observability and tracing for debugging and optimization

##  Architecture

### Core Components

1. **Document Processing**
   - PowerPoint (.pptx) text extraction
   - Document chunking with overlap
   - Semantic embedding generation
   - FAISS vector indexing

2. **Language Models**
   - Cohere Command-R for generation
   - Structured output parsing
   - Multi-tool routing system

3. **Quality Assessment**
   - Document relevance grader
   - Hallucination detection
   - Answer quality evaluation
   - Taxonomy classification

4. **Workflow Engine**
   - LangGraph state management
   - Conditional routing logic
   - Retry mechanisms
   - Performance tracking

## Prerequisites

- Python 3.8+
- Valid Cohere API key
- LangSmith API key (optional, for tracing)
- PowerPoint files containing cybersecurity knowledge base

##  Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cybersecurity-ai-offer-generator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export COHERE_API_KEY="your_cohere_api_key"
export LANGCHAIN_API_KEY="your_langsmith_api_key"  # Optional
export LANGCHAIN_TRACING_V2="true"  # Optional
```

4. **Update document path**
Edit the `drive_path` variable in `app.py` to point to your PowerPoint files:
```python
drive_path = r"C:\path\to\your\documents"
```

##  Dependencies

### Core Libraries
```
flask>=2.0.0
flask-cors>=4.0.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-cohere>=0.1.0
langgraph>=0.0.40
langsmith>=0.1.0
```

### AI/ML Libraries
```
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.21.0
```

### Document Processing
```
python-pptx>=0.6.0
pydantic>=2.0.0
```

### Utilities
```
psutil>=5.8.0
typing-extensions>=4.0.0
```

##  Usage

### Starting the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### Generate Cybersecurity Offer
```http
POST /generate
Content-Type: application/json

{
    "question": "Générer une offre pour la détection des menaces avec IA"
}
```

**Response:**
```json
{
    "steps": [...],
    "final_result": "Generated offer content...",
    "metrics": {
        "quality": {
            "overall_score": 0.85,
            "grade": "B",
            "detailed_scores": {...}
        },
        "processing_time": 2.34,
        "cost": 0.0023,
        "confidence_score": 87
    }
}
```

#### Health Check
```http
GET /health
```

#### Component Testing
```http
GET /test-components
```

#### Taxonomy Information
```http
GET /taxonomy
POST /taxonomy/classify
```

#### Document Statistics
```http
GET /documents/stats
```

##  Offer Structure

Generated offers include the following sections:

1. **RÉSUMÉ EXÉCUTIF** - Executive summary of the AI cybersecurity solution
2. **SOLUTION TECHNIQUE** - Detailed technical approach using AI
3. **PLAN DE MISE EN ŒUVRE** - Step-by-step deployment strategy
4. **CALENDRIER** - Project phases and milestones
5. **COMPOSITION DE L'ÉQUIPE** - Required expertise and roles
6. **RÉSULTATS ATTENDUS** - Measurable benefits and ROI
7. **STRUCTURE DE PRIX** - Cost breakdown and investment details

## Taxonomy Categories

### Cybersecurity
- **Threat Detection**: SIEM, SOC, threat hunting, anomaly detection
- **Vulnerability Management**: Penetration testing, security audit, risk assessment
- **Compliance**: GDPR, ISO27001, SOX, PCI DSS
- **Incident Response**: Forensics, recovery, investigation, containment
- **Identity Management**: SSO, MFA, privileged access, identity governance

### AI Solutions
- **Machine Learning**: Supervised, unsupervised, reinforcement, deep learning
- **NLP**: Chatbots, document processing, sentiment analysis, language models
- **Computer Vision**: Object detection, image classification, facial recognition
- **Predictive Analytics**: Forecasting, anomaly detection, pattern recognition

##  Quality Metrics

The system calculates comprehensive quality metrics:

- **Precision**: Technical term coverage and accuracy
- **Consistency**: Structural quality and coherence
- **Completeness**: Required section coverage
- **Relevance**: Question alignment and focus
- **Overall Score**: Weighted combination (A-F grading)

##  Workflow Pipeline

1. **Question Routing**: Determines data source (vectorstore vs web search)
2. **Document Retrieval**: Semantic search for relevant content
3. **Document Grading**: Filters irrelevant documents
4. **Generation**: Creates cybersecurity offer
5. **Quality Control**: Checks for hallucinations and answer quality
6. **Metrics Calculation**: Comprehensive performance assessment

##  Troubleshooting

### Common Issues

**No documents loaded:**
- Check that `drive_path` points to valid PowerPoint files
- Ensure files are not password-protected or corrupted

**API errors:**
- Verify Cohere API key is valid and has sufficient credits
- Check network connectivity

**Memory issues:**
- Reduce `chunk_size` in document splitter
- Process fewer documents at once

**Generation quality:**
- Add more relevant documents to knowledge base
- Adjust temperature settings
- Review prompt engineering

### Debug Endpoints

- `/health` - Check component status
- `/test-components` - Test individual components
- `/documents/stats` - Document processing statistics

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `COHERE_API_KEY` | Cohere API key for LLM | Yes |
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing | No |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | No |

### Customization

**Document Path:**
```python
drive_path = r"C:\path\to\your\documents"
```

**Model Configuration:**
```python
base_llm = ChatCohere(
    model="command-r",
    temperature=0,  # Adjust for creativity vs consistency
    cohere_api_key=os.environ["COHERE_API_KEY"]
)
```

**Embedding Model:**
```python
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
```

##  Performance Optimization

### Tips for Better Performance

1. **Document Optimization**
   - Keep PowerPoint files under 50MB each
   - Use clear, structured content
   - Remove unnecessary images and animations

2. **Memory Management**
   - Monitor memory usage via `/health` endpoint
   - Restart application if memory usage exceeds 2GB
   - Consider document pagination for large datasets

3. **Quality Improvement**
   - Add domain-specific documents to knowledge base
   - Fine-tune chunk sizes based on document structure
   - Adjust grading thresholds for specific use cases

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



##  Future Enhancements

- [ ] Multi-language support (English, Spanish)
- [ ] PDF document processing
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Custom taxonomy management
- [ ] Export to multiple formats (PDF, Word, PowerPoint)
- [ ] Integration with CRM systems
- [ ] Multi-tenant support

---

**DEMO**
<img width="1853" height="822" alt="image" src="https://github.com/user-attachments/assets/14c92cd3-c32e-4fd4-8e45-555c050ff671" />
<img width="1586" height="843" alt="image" src="https://github.com/user-attachments/assets/d2cf70ad-338f-4715-b2e0-6ff5f8362931" />
<img width="1614" height="659" alt="image" src="https://github.com/user-attachments/assets/f95a1c3c-ee75-4d17-8994-598d9387f061" />
<img width="1567" height="864" alt="image" src="https://github.com/user-attachments/assets/dfa83d35-21d1-4fc1-baf3-99b1f0bc45b5" />
<img width="1780" height="830" alt="image" src="https://github.com/user-attachments/assets/ca55b5e4-8fa8-4f44-9cfc-420122939c63" />
<img width="1856" height="837" alt="image" src="https://github.com/user-attachments/assets/4b0062ee-2ebb-4a9f-be98-35a2a15d002b" />
<img width="1913" height="777" alt="image" src="https://github.com/user-attachments/assets/1f616bbd-3ffe-4176-a3a4-9f9ff8f013ba" />
