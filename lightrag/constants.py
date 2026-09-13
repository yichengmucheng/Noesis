"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

# Default values for server settings
DEFAULT_WOKERS = 10
DEFAULT_MAX_GRAPH_NODES = 3000

# Default values for extraction settings
DEFAULT_SUMMARY_LANGUAGE = "Chinese"  # Default language for document processing
DEFAULT_MAX_GLEANING = 2

# Number of description fragments to trigger LLM summary
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8
# Max description token size to trigger LLM summary
DEFAULT_SUMMARY_MAX_TOKENS = 1200
# Recommended LLM summary output length in tokens
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
# Maximum token size sent to LLM for summary
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000
# Default entities to extract if ENTITY_TYPES is not specified in .env

# DEFAULT_ENTITY_TYPES = [ "Person", "Organization", "Location", "Event", "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject", ]
DEFAULT_ENTITY_TYPES = [
    "设备",        # Person
    "部件",        # Organization
    "故障定义",        # Location
    "一层原因",        # Event
    "二层原因",        # Event
    "三层原因",        # Event
    "四层原因",        # Event
    "问题点",        # Event
    "措施",        # Concept
    "验证试验",        # Method
    "文件",        # Content
    "人员",        # Data
    "部门",     # NaturalObject
    "效果",    # NaturalObject
]

# Separator for graph fields
GRAPH_FIELD_SEP = "<SEP>"

# Query and retrieval configuration defaults
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 30
DEFAULT_MAX_ENTITY_TOKENS = 6000
DEFAULT_MAX_RELATION_TOKENS = 8000
DEFAULT_MAX_TOTAL_TOKENS = 26000
DEFAULT_COSINE_THRESHOLD = 0.15
DEFAULT_RELATED_CHUNK_NUMBER = 5
DEFAULT_KG_CHUNK_PICK_METHOD = "VECTOR"
# Deprated: history message have negtive effect on query performance
DEFAULT_HISTORY_TURNS = 0

# Rerank configuration defaults
DEFAULT_MIN_RERANK_SCORE = 0.0
DEFAULT_RERANK_BINDING = "cohere"

# File path configuration for vector and graph database(Should not be changed, used in Milvus Schema)
DEFAULT_MAX_FILE_PATH_LENGTH = 28768

# Default temperature for LLM
DEFAULT_TEMPERATURE = 0.3

# Async configuration defaults
DEFAULT_MAX_ASYNC = 1  # Default maximum async operations
DEFAULT_MAX_PARALLEL_INSERT = 1  # Default maximum parallel insert operations

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 3  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 5  # Default batch size for embedding computations

# Gunicorn worker timeout
DEFAULT_TIMEOUT = 300

# Default llm and embedding timeout
DEFAULT_LLM_TIMEOUT = 360
DEFAULT_EMBEDDING_TIMEOUT = 120

# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "lightrag.log"  # Default log filename

# Ollama server configuration defaults
DEFAULT_OLLAMA_MODEL_NAME = "lightrag"
DEFAULT_OLLAMA_MODEL_TAG = "latest"
DEFAULT_OLLAMA_MODEL_SIZE = 7365960935
DEFAULT_OLLAMA_CREATED_AT = "2024-01-15T00:00:00Z"
DEFAULT_OLLAMA_DIGEST = "sha256:lightrag"
