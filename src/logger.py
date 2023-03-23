import src.config as config
import logging


def get_logger():
    level = config.LOG_LEVEL.lower()
    log_level = logging.INFO

    if level == "debug":
        log_level = logging.DEBUG
    elif level == "info":
        log_level = logging.INFO
    elif level == "warning":
        log_level = logging.WARNING
    elif level == "error":
        log_level = logging.ERROR
    else:
        log_level = logging.CRITICAL

    logging.basicConfig(
            filename='./logs/application.log',
            level=log_level,
            filemode='a', # w =  overwrite, a = append
            format='%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
    )    

    return logging.getLogger()
