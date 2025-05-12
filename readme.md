# AI Tagging Service

A production-grade REST API service for classifying and tagging business terms against Preferred Business Terms (PBT) using AI.

## Features

- **AI-powered Classification**: Automatically tag business terms using advanced AI techniques
- **Multiple Classification Methods**:
  - Embedding-based similarity matching
  - LLM classification
  - LangGraph agent with reasoning
- **CDM Support**: Integrate with Conceptual Data Model (CDM) categorization
- **Synonym Generation**: Automatically generate synonyms for better term matching
- **Confidence Scoring**: Evaluate the confidence of matches
- **Production Features**:
  - REST API with FastAPI
  - Rate limiting
  - Request logging and monitoring
  - Health checks
  - Docker support
  - Production-grade configuration

## Architecture

The application is built on a modern, maintainable architecture:

- **FastAPI**: Modern, high-performance API framework
- **Langchain & LangGraph**: Framework for AI agents and LLM integration
- **ChromaDB**: Vector database for semantic search
- **Azure OpenAI**: AI models for classification and embeddings
- **Pydantic**: Data validation and settings management

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)
- Azure OpenAI API credentials

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-tagging-service.git
   cd ai-tagging-service
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp env/config.env.example env/config.env
   # Edit env/config.env with your settings
   ```

5. Prepare your PBT data:
   Place your PBT data CSV file in the `data` directory. The CSV should have columns:
   - `id`: Unique identifier
   - `PBT_NAME`: Name of the business term
   - `PBT_DEFINITION`: Definition of the business term
   - `CDM`: Conceptual Data Model category (optional)

### Running with Docker

1. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

2. The API will be available at http://localhost:8000

### Running Locally

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Classification Endpoints

- `POST /api/v1/classification/classify`: Classify a single business term
- `POST /api/v1/classification/batch`: Classify multiple business terms
- `POST /api/v1/classification/load-pbt`: Load PBT data from CSV
- `GET /api/v1/classification/pbt/{pbt_id}`: Get a PBT by ID
- `GET /api/v1/classification/statistics`: Get PBT statistics

### Health Check Endpoints

- `GET /api/v1/health`: Detailed health check
- `GET /api/v1/health/ready`: Readiness check
- `GET /api/v1/health/live`: Liveness check

## Example Request

```bash
curl -X POST http://localhost:8000/api/v1/classification/classify \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Account Number",
    "description": "Unique identifier assigned to a customer account",
    "method": "agent"
  }'
```

## Directory Structure

```
ai_tagging_service/
├── app/
│   ├── api/
│   │   ├── deps.py                  # Dependency injection
│   │   ├── endpoints/
│   │   │   ├── classification.py    # Classification endpoints
│   │   │   └── health.py            # Health check endpoints
│   ├── config/
│   │   ├── environment.py           # Environment variables
│   │   ├── logging_config.py        # Logging configuration
│   │   └── settings.py              # Application settings
│   ├── core/
│   │   ├── auth/
│   │   │   └── auth_helper.py       # Authentication helpers
│   │   ├── models/
│   │   │   └── pbt.py               # PBT data models
│   │   ├── services/
│   │   │   ├── classification.py    # Classification service
│   │   │   ├── confidence.py        # Confidence evaluation
│   │   │   ├── embeddings.py        # Embeddings service
│   │   │   └── pbt_manager.py       # PBT data manager
│   │   └── vector_store/
│   │       └── chroma_store.py      # ChromaDB implementation
│   ├── utils/
│   │   └── helpers.py               # Utility functions
│   └── main.py                      # Application entry point
├── data/                            # Data files
│   └── pbt_data.csv                 # PBT data CSV
├── env/                             # Environment files
│   └── config.env                   # Configuration
├── logs/                            # Log files
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose configuration
└── requirements.txt                 # Python dependencies
```

## Development

### Testing

```bash
pytest
```

### Code Formatting

```bash
black app tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.