# import os   #logging file
# import sys
# import logging

# logging_str = "[%(asctime)s: %(levelname)s:%(module)s:%(message)s]"
# log_dir = "logs"
# log_filepath = os.path.join(log_dir,"logging.log")
# os.makedirs(log_dir, exist_ok = True)

# logging.basicConfig(
#     format=logging_str, level=logging.INFO,
#     handlers = [
#         logging.FileHandler(log_filepath),
#         logging.StreamHandler(sys.stdout) #to see output(log_filepath) in terminal
#     ]
# )

# logger = logging.getLogger("Spam_Detection-Project")
import logging
import os
from datetime import datetime

log_file_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = os.path.join("logs", log_file_name)
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)