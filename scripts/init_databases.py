#!/usr/bin/env python
"""
Initialize all databases for CDSS.
Run: python scripts/init_databases.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import get_settings
from src.database.session import init_db
from src.knowledge_graph.schema import init_knowledge_graph
from src.vector_store.schema import init_vector_store


def main():
    """Initialize all databases"""
    settings = get_settings()
    
    logger.info("=" * 50)
    logger.info("CDSS Database Initialization")
    logger.info("=" * 50)
    
    # 1. PostgreSQL
    logger.info("\n1. Initializing PostgreSQL...")
    try:
        init_db()
        logger.success("PostgreSQL tables created")
    except Exception as e:
        logger.error(f"PostgreSQL init failed: {e}")
        logger.info("Make sure PostgreSQL is running: docker-compose up -d postgres")
    
    # 2. Neo4j Knowledge Graph
    logger.info("\n2. Initializing Neo4j Knowledge Graph...")
    try:
        init_knowledge_graph(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        logger.success("Neo4j schema and data populated")
    except Exception as e:
        logger.error(f"Neo4j init failed: {e}")
        logger.info("Make sure Neo4j is running: docker-compose up -d neo4j")
    
    # 3. Weaviate Vector Store
    logger.info("\n3. Initializing Weaviate Vector Store...")
    try:
        init_vector_store(url=settings.weaviate_url)
        logger.success("Weaviate schema created")
    except Exception as e:
        logger.error(f"Weaviate init failed: {e}")
        logger.info("Make sure Weaviate is running: docker-compose up -d weaviate")
    
    logger.info("\n" + "=" * 50)
    logger.info("Database initialization complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
