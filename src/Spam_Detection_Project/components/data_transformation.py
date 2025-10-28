import os
import re
from typing import Optional

from src.Spam_Detection_Project import logger
import pandas as pd
from src.Spam_Detection_Project.entity.config_entity import (DataTransformationConfig)
from sklearn.model_selection import train_test_split


class DataTransformation:
    """Performs feature engineering on the raw CSV and produces train/test splits.

    The implementation adapts the feature functions from the provided notebook and
    saves three files into `self.config.root_dir`:
      - processed_data.csv  (full dataset after feature creation)
      - Train.csv
      - Test.csv

    Methods:
      - _feature functions: small helpers (phone/link/keywords/etc.)
      - process_and_split(): run the full pipeline
      - train_test_sepeartion(): kept for backward compatibility and calls process_and_split
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # ----- Feature helper functions (adapted from the notebook) -----
    def _has_phone_number(self, text: object) -> int:
        pattern = r"\b[\+]?[0-9][0-9\-]{9,}\b"
        return int(bool(re.search(pattern, str(text))))

    def _has_link(self, text: object) -> int:
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return int(bool(re.search(pattern, str(text))))

    def _keywords(self, text: object) -> int:
        keywords = [
            "मुफ़्त", "दावा", "जीत", "नकद", "प्रस्ताव", "सीमित", "पुरस्कार", "पैसा",
            "मौका", "लिखकर", "लाख", "stop", "हज़ार", "claim", "free",
            "urgent", "act now", "winner",
            'फ्री', 'जल्दी', 'लिमिटेड', 'विजेता', 'इनाम', 'ऑफर', 'कॉल', 'क्लिक', 'लकी',
            'खरीदें', 'बधाई', 'शीघ्र'
        ]
        txt = str(text).lower()
        for kw in keywords:
            if kw.lower() in txt:
                return 1
        return 0

    def _special_characters(self, text: object) -> int:
        # matches currency amounts or punctuation characters
        pattern = r"(?:₹|RS|INR|\$)\s*\d+(?:,\d+)*(?:\.\d{2})?|[!@#$%^&*(),.?\":{}|<>]"
        return int(bool(re.search(pattern, str(text))))

    def _cash_amount(self, text: object) -> int:
        cash_keywords = ["1 लाख", "दस लाख", "1 हज़ार", "दस हज़ार", "करोड़", "दस करोड़", "मिलियन", "बिलियन", "सौ", "लाख", "हज़ार"]
        txt = str(text)
        for c in cash_keywords:
            if c in txt:
                return 1
        return 0

    def _length_sms(self, text: object) -> int:
        return len(str(text))

    def _sms_number(self, text: object) -> int:
        pattern = r"\b[5-9]\d{4,5}\b"
        return int(bool(re.search(pattern, str(text))))

    def _word_count(self, text: object) -> int:
        return len(str(text).split())

    # ----- Main pipeline -----
    def process_and_split(self, test_size: float = 0.2, random_state: int = 42) -> Optional[tuple]:
        """Create features, save a processed CSV, and write Train/Test CSVs.

        Returns:
          (train_df, test_df) on success, otherwise None.
        """
        try:
            # Read input data
            data_path = getattr(self.config, "data_path", None) or getattr(self.config, "data_path", None)
            if not data_path or not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                return None

            df = pd.read_csv(data_path)

            # Safety: ensure message column exists
            message_col = None
            for candidate in ["Message", "message", "text", "Text"]:
                if candidate in df.columns:
                    message_col = candidate
                    break

            # Create features if we have a message column
            if message_col is not None:
                df["contains_phone_number"] = df[message_col].apply(self._has_phone_number)
                df["contains_URL_link"] = df[message_col].apply(self._has_link)
                df["Keywords"] = df[message_col].apply(self._keywords)
                df["Special_Characters"] = df[message_col].apply(self._special_characters)
                df["Amount"] = df[message_col].apply(self._cash_amount)
                df["Length"] = df[message_col].apply(self._length_sms)
                df["SMS_Number"] = df[message_col].apply(self._sms_number)
                df["word_count"] = df[message_col].apply(self._word_count)
            else:
                logger.warning("No message/text column found — skipping text feature creation")

            # Create a numeric label column if Category exists
            if "Category" in df.columns:
                # Notebook used 1 for ham, 0 for spam; keep same mapping
                df["label_numeric"] = df["Category"].apply(lambda x: 1 if str(x).lower() == "ham" else 0)

            # Ensure output directory exists
            os.makedirs(self.config.root_dir, exist_ok=True)

            processed_path = os.path.join(self.config.root_dir, "processed_data.csv")
            df.to_csv(processed_path, index=False)
            print(df.info())
            logger.info(f"Saved processed data to: {processed_path}")

            # Prepare for split
            if "label_numeric" in df.columns:
                stratify_col = df["label_numeric"]
            else:
                stratify_col = None

            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=stratify_col
            )

            train_path = os.path.join(self.config.root_dir, "Train.csv")
            test_path = os.path.join(self.config.root_dir, "Test.csv")
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("The split has been successful!")
            logger.info(f"Train shape: {train_df.shape}")
            logger.info(f"Test shape: {test_df.shape}")
            print(train_df.shape)
            print(test_df.shape)

            return train_df, test_df

        except Exception as e:
            logger.exception(e)
            raise

    # keep old name for backward-compatibility
    def train_test_sepeartion(self):
        return self.process_and_split()