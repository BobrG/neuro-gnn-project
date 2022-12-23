import logging

def get_logger(logger_name, logfile=None, output=None):
    format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.CRITICAL)
    stream_handler.setFormatter(format)
    logger.addHandler(stream_handler)
        
    if logfile is not None:
        handler = logging.FileHandler(logfile)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(format)
        logger.addHandler(handler)
    
    return logger