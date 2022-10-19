import logging

def get_logger(logger_name, logfile):
    format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
        
    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(format)
    logger.addHandler(handler)
    
    return logger