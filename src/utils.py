import logging

def setup_logging():
    """Configure logging."""
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()