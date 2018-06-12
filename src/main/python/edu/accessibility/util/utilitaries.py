import configparser
import datetime
import json
import logging
import logging.config
import os

import time

WGS_84 = {'init': 'EPSG:4326'}
PSEUDO_MERCATOR = {'init': 'EPSG:3857'}


def getConfigurationProperties(section="WFS_CONFIG"):
    config = configparser.ConfigParser()
    configurationPath = os.path.join(os.getcwd(), "src", "main", "resources", "configuration.properties")
    config.read(configurationPath)
    return config[section]


def getFormattedDatetime(timemilis=time.time(), format='%Y-%m-%d %H:%M:%S'):
    formattedDatetime = datetime.datetime.fromtimestamp(timemilis).strftime(format)
    return formattedDatetime


def getTimestampFromString(date_string, format='%Y-%m-%d %H:%M:%S'):
    formattedDatetime = datetime.datetime.strptime(date_string, format)
    # return calendar.timegm(formattedDatetime.utctimetuple())
    return formattedDatetime.timestamp()


def timeDifference(startTime, endTime):
    totalTime = (endTime - startTime) / 60  # min
    return totalTime


def dgl_timer(func):
    def func_wrapper(*args, **kwargs):
        functionName = func.__name__
        startTime = time.time()
        Logger.getInstance().info("%s Start Time: %s" % (functionName, getFormattedDatetime(timemilis=startTime)))

        ###############################
        returns = func(*args, **kwargs)
        ###############################

        endTime = time.time()
        Logger.getInstance().info("%s End Time: %s" % (functionName, getFormattedDatetime(timemilis=endTime)))

        totalTime = timeDifference(startTime, endTime)
        Logger.getInstance().info("%s Total Time: %s m" % (functionName, totalTime))

        return returns

    return func_wrapper

def parallel_job_print(msg, msg_args):
    """ Display the message on stout or stderr depending on verbosity
    """
    # XXX: Not using the logger framework: need to
    # learn to use logger better.
    # if not self.verbose:
    #     return
    # if self.verbose < 50:
    #     writer = sys.stderr.write
    # else:
    #     writer = sys.stdout.write
    msg = msg % msg_args
    self = "Parallel(n_jobs=%s)" % getConfigurationProperties(section="PARALLELIZATION")["jobs"]
    # writer('[%s]: %s\n' % (self, msg))
    Logger.getInstance().info('[%s]: %s' % (self, msg))

class FileActions:
    def readJson(self, url):
        """
        Read a json file
        :param url: URL for the Json file
        :return: json dictionary data
        """
        with open(url) as f:
            data = json.load(f)
        return data

    def createFolder(self, folderPath):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

    def writeFile(self, folderPath, filename, data):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        fileURL = os.path.join(folderPath, filename)

        with open(fileURL, 'w+') as outfile:
            json.dump(data, outfile, sort_keys=True)

        return fileURL

    def createFile(self, folderPath, filename):
        fileURL = os.path.join(folderPath, filename)
        with open(fileURL, 'w+') as outfile:
            outfile.close()


class GeneralLogger:
    def __init__(self, loggerName, outputFolder, prefix=""):
        self.logger = self._createLogger(loggerName=loggerName)
        self.handler = self._createLogFileHandler(outputFolder=outputFolder, prefix=prefix)

        self.logger.addHandler(self.handler)

    def _createLogger(self, loggerName):
        configurationPath = os.path.join(os.getcwd(), "src", "main", "resources", "logging.properties")
        logging.config.fileConfig(configurationPath)
        # create logger
        logger = logging.getLogger(loggerName)
        return logger

    def _createLogFileHandler(self, outputFolder, prefix):
        log_filename = prefix + "_log_%s.log" % getFormattedDatetime(
            timemilis=time.time(),
            format='%Y_%m_%d__%H_%M_%S'
        )
        logs_folder = os.path.join(outputFolder, "logs")
        FileActions().createFile(logs_folder, log_filename)

        fileHandler = logging.FileHandler(logs_folder + os.sep + log_filename, 'w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fileHandler.setFormatter(formatter)
        return fileHandler

    def getLogger(self):
        return self.logger

    def clean(self):
        self.logger.removeHandler(self.handler)
        self.handler.flush()
        self.handler.close()


class Logger:
    __instance = None

    def __init__(self):
        raise Exception("Instances must be constructed with Logger.getInstance()")

    @staticmethod
    def configureLogger(outputFolder, prefix):
        if Logger.__instance:
            Logger.__instance.clean()
        Logger.__instance = GeneralLogger("OTPAnalyzer LOGGER", outputFolder=outputFolder, prefix=prefix)

    @staticmethod
    def getInstance():
        if not Logger.__instance:
            raise Exception("Call configureLogger before call getInstance()")
        # "application" code
        # Logger.instance.debug("debug message")
        # Logger.instance.info("info message")
        # Logger.instance.warn("warn message")
        # Logger.instance.error("error message")
        # Logger.instance.critical("critical message")
        return Logger.__instance.getLogger()

    def __del__(self):
        Logger.__instance.clean()


class Counter:
    maxPlansToProcess = 0
    generalCounter = 0

    @staticmethod
    def getPercentage():
        if Counter.maxPlansToProcess == 0:
            return 0
        return round(Counter.generalCounter/Counter.maxPlansToProcess, 2) * 100