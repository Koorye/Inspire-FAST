import datetime
import logging
import uuid


def reset_logging():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


class Logger:
    def __init__(self, log_file, mode='w', name=None, log_level='info'):
        if name is None:
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(uuid.uuid4())
        
        self.logger = logging.getLogger(name)
        # console handler and file handler
        ch = logging.StreamHandler()
        fh = logging.FileHandler(log_file, mode=mode)
        # formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add handlers
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        if log_level == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif log_level == 'info':
            self.logger.setLevel(logging.INFO)
        elif log_level == 'warning':
            self.logger.setLevel(logging.WARNING)
        elif log_level == 'error':
            self.logger.setLevel(logging.ERROR)
        elif log_level == 'critical':
            self.logger.setLevel(logging.CRITICAL)
        else:
            raise ValueError(f"Invalid log level: {log_level}")

    def debug(self, msg):
        self.logger.debug(msg)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def critical(self, msg):
        self.logger.critical(msg)

    def infos(self, title, msgs, sort=True):
        self.info('-' * 40)
        self.info(title + ':')

        if isinstance(msgs, dict):
            if sort:
                msgs = {k: v for k, v in sorted(msgs.items(), key=lambda x: x[0])}
            for k, v in msgs.items():
                self.info(f"{k}: {v}")
        else:
            if sort:
                msgs = sorted(msgs)
            for msg in msgs:
                self.info(msg)
        
        self.info('-' * 40)
