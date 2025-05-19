from motor.motor_asyncio import AsyncIOMotorClient
from ..core.config import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect_to_database(cls):
        try:
            cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
            cls.db = cls.client[settings.DATABASE_NAME]
            logger.info("Connected to MongoDB!")
        except Exception as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise

    @classmethod
    async def close_database_connection(cls):
        if cls.client is not None:
            cls.client.close()
            logger.info("Closed MongoDB connection!")

    @classmethod
    def get_database(cls):
        if cls.db is None:
            raise Exception("Database not initialized. Call connect_to_database first.")
        return cls.db

    @classmethod
    def get_collection(cls, collection_name: str):
        db = cls.get_database()
        return db[collection_name]

# Create database instance
mongodb = MongoDB() 