import os
from contextlib import contextmanager
from threading import Lock
from typing import Optional, Generator
import logging

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConfigurationError, ConnectionFailure

# Suppress verbose pymongo debug logs
logging.getLogger("pymongo").setLevel(logging.WARNING)

import sys
import os
# Ensure project root is on sys.path so local modules (logging_config) import correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_config import logger


class MongoDBConfig:
    @classmethod
    def get_uri(cls) -> str:
        uri = os.getenv("MONGODB_CONNECTION_STRING")
        if not uri:
            raise ValueError(
                "MONGODB_CONNECTION_STRING environment variable is not set. "
                "Cannot connect to MongoDB."
            )
        return uri

    @classmethod
    def get_database_name(cls) -> str:
        return os.getenv("MONGODB_DATABASE", "pac_grader")

    @classmethod
    def get_client_kwargs(cls) -> dict:
        return {
            "connectTimeoutMS": 15000,
            "serverSelectionTimeoutMS": 10000,
            "socketTimeoutMS": 20000,
            "retryWrites": True,
            "retryReads": True,
            # "tls": True,               # usually automatic with +srv
            # "tlsAllowInvalidCertificates": False,
        }


class MongoDBConnection:
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None
    _lock = Lock()  # protects initialization

    @classmethod
    def _initialize(cls) -> None:
        if cls._client is not None:
            return

        with cls._lock:
            if cls._client is not None:  # double-checked locking
                return

            logger.info("Initializing MongoDB connection...")

            uri = MongoDBConfig.get_uri()
            db_name = MongoDBConfig.get_database_name()
            kwargs = MongoDBConfig.get_client_kwargs()

            def _try_connect(test_uri: str):
                c = MongoClient(test_uri, **kwargs)
                # Force connection check
                c.admin.command("ping")
                return c
            # Primary attempt using configured URI
            try:
                cls._client = _try_connect(uri)
                cls._db = cls._client.get_database(db_name)
                logger.info("MongoDB connection established", extra={"database": db_name})
            except (ConfigurationError, ConnectionFailure) as e:
                logger.warning("Primary MongoDB connection failed", exc_info=True)

                # Fallback: allow providing a non-SRV (mongodb://) URI via env
                std_uri = os.getenv("MONGODB_STANDARD_URI") or os.getenv("MONGODB_DIRECT_URI")
                if std_uri:
                    try:
                        logger.info("Attempting non-SRV MongoDB URI from environment variable")
                        cls._client = _try_connect(std_uri)
                        cls._db = cls._client.get_database(db_name)
                        logger.info("MongoDB connected via non-SRV URI")
                    except Exception:
                        logger.error("Non-SRV URI attempt failed", exc_info=True)
                        cls._client = None
                        cls._db = None

                if cls._client is None:
                    msg = (
                        f"MongoDB connection failed: {e}. "
                        "If your network blocks SRV DNS queries, set MONGODB_STANDARD_URI to a mongodb:// URI with host:port list and retry."
                    )
                    logger.error(msg)
                    raise ConnectionError(msg) from e
            except Exception:
                logger.error("Unexpected error during MongoDB initialization", exc_info=True)
                raise

    @classmethod
    def get_client(cls) -> MongoClient:
        cls._initialize()
        if cls._client is None:
            raise RuntimeError("MongoDB client not initialized")
        return cls._client

    @classmethod
    def get_db(cls) -> Database:
        cls._initialize()
        if cls._db is None:
            raise RuntimeError("MongoDB database not initialized")
        return cls._db

    @classmethod
    def get_collection(cls, name: str = "pac_questions") -> Collection:
        return cls.get_db()[name]

    @classmethod
    def close(cls) -> None:
        if cls._client:
            try:
                cls._client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.warning(f"Error while closing MongoDB connection: {e}")
            finally:
                cls._client = None
                cls._db = None

    @classmethod
    @contextmanager
    def context(cls) -> Generator[Database, None, None]:
        cls._initialize()
        try:
            yield cls.get_db()
        finally:
            pass


# Convenience exports
get_db = MongoDBConnection.get_db
get_collection = MongoDBConnection.get_collection
def get_questions_collection():
    return MongoDBConnection.get_collection("pac_questions")
close = MongoDBConnection.close