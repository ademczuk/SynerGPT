# logging_config.py
import logging

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

def init_logger():
    logging.basicConfig(
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        filename='app.log',
        filemode='w'
    )