import os
import sys
import pymongo
from logging import Logger
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from src.Spam_Detection_Project.logger.logger import logging
from src.Spam_Detection_Project.exception.exception import CustomException
import certifi

MONGO_DB_URL = os.environ.get("MONGO_DB_URL")
# Do not raise at import time — allow the app to start even if Mongo is not configured.
# The Spam_Detection_MongoDB class will check and disable DB operations if the URL is missing.
if not MONGO_DB_URL:
    logging.warning("MONGO_DB_URL environment variable not set — MongoDB functionality will be disabled until configured.")

ca = certifi.where()


class Spam_Detection_MongoDB:
    def __init__(self, logger: Logger, db_name: str = 'Spam_Detection_DB', collection_name: str = 'predictions'):
        self.logger = logger
        self.db_name = db_name
        self.collection_name = collection_name
        try:
            if not MONGO_DB_URL:
                # Graceful degradation: don't attempt to connect when URL is missing
                self.client = None
                self.db = None
                self.collection = None
                self.logger.warning("MONGO_DB_URL not provided; MongoDB client not initialized.")
                return

            self.client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.db = self.client.get_database(db_name)
            self.collection = self.db.get_collection(collection_name)
            self.logger.info(f"Connected to MongoDB successfully. DB: {db_name}, Collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error connecting to MongoDB: {e}")
            raise e

    def insert_prediction(self, data: dict):
        try:
            if self.collection is None:
                self.logger.warning("insert_prediction called but MongoDB is not configured. Skipping insert.")
                return None

            data['timestamp'] = datetime.utcnow()
            result = self.collection.insert_one(data)
            self.logger.info(f"Inserted document with id: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            self.logger.error(f"Error inserting document: {e}")
            raise e

    def fetch_predictions(self, query: dict):
        try:
            if self.collection is None:
                self.logger.warning("fetch_predictions called but MongoDB is not configured. Returning empty list.")
                return []

            results = self.collection.find(query)
            return list(results)
        except Exception as e:
            self.logger.error(f"Error fetching documents: {e}")
            raise e
    
    def csv_to_json_converter(self, file_path: str) -> list:
        """Convert a CSV file to a JSON-like dictionary format suitable for MongoDB insertion."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            json_data = df.to_dict(orient='records')
            self.logger.info(f"Converted CSV to JSON with {len(json_data)} records from {file_path}")
            return json_data
        except Exception as e:
            self.logger.error(f"Error converting CSV to JSON: {e}")
            raise CustomException(e, sys)
    
    def insert_many_records(self, records: list, add_timestamp: bool = True) -> list:
        """Insert multiple records (bulk insert).
        
        Args:
            records: List of dictionaries to insert
            add_timestamp: If True, add 'timestamp' field to each record
            
        Returns:
            List of inserted document IDs
        """
        try:
            if not records:
                self.logger.warning("No records to insert")
                return []
            
            if self.collection is None:
                self.logger.warning("insert_many_records called but MongoDB is not configured. Skipping insert.")
                return []

            if add_timestamp:
                for record in records:
                    if 'timestamp' not in record:
                        record['timestamp'] = datetime.utcnow()

            result = self.collection.insert_many(records)
            self.logger.info(f"Inserted {len(result.inserted_ids)} documents successfully")
            return result.inserted_ids
        except Exception as e:
            self.logger.error(f"Error inserting records: {e}")
            raise CustomException(e, sys)
    
    def push_csv_to_mongodb(self, file_path: str) -> int:
        """Convert CSV file and push all records to MongoDB in one call.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Number of records inserted
        """
        try:
            records = self.csv_to_json_converter(file_path)
            inserted_ids = self.insert_many_records(records, add_timestamp=True)
            self.logger.info(f"Successfully pushed {len(inserted_ids)} records from {file_path} to MongoDB")
            return len(inserted_ids)
        except Exception as e:
            self.logger.error(f"Error pushing CSV to MongoDB: {e}")
            raise CustomException(e, sys)


# ====== USAGE EXAMPLE ======
if __name__ == "__main__":
    """
    Example: Push processed_data.csv to MongoDB
    """
    try:
        # Initialize MongoDB connector
        mongo_db = Spam_Detection_MongoDB(
            logger=logging,
            db_name='Spam_Detection_DB',
            collection_name='predictions'
        )
        # Push processed_data.csv to MongoDB
        csv_file_path = "artifacts/data_transformation/processed_data.csv"
        num_inserted = mongo_db.push_csv_to_mongodb(csv_file_path)
        print(f"✓ Successfully pushed {num_inserted} records to MongoDB")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)