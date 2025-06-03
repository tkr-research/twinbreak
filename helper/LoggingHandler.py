from __future__ import annotations

import os
from datetime import datetime

from helper.FileHandler import FileHandler


class LoggingHandler:
    """Central logging class. Defines file names and basic logging behavior."""

    PRINT = True

    # region Static state variables
    __LOG_FILE_LOCATION_PREFIX: str = '/LogFiles'
    # Can be modified by update_log_file_name
    __LOG_FILE_NAME: str = 'log'
    __LOG_FILE_EXTENSION: str = '.txt'
    # Initialized
    __INITIALIZED_FILENAME: str | None = None

    # endregion

    # region Log File
    @staticmethod
    def get_log_file() -> str:
        if LoggingHandler.__INITIALIZED_FILENAME is None:
            FileHandler.ensure_dir_exists(LoggingHandler.__LOG_FILE_LOCATION_PREFIX)
            LoggingHandler.__INITIALIZED_FILENAME = LoggingHandler.__LOG_FILE_LOCATION_PREFIX + \
                                                    "/" + LoggingHandler.__get_new_log_file_name(
                LoggingHandler.__LOG_FILE_LOCATION_PREFIX
                )
        return LoggingHandler.__INITIALIZED_FILENAME

    @staticmethod
    def __get_new_log_file_name(folder: str) -> str:
        file_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        log_identifier = 0
        for logfile in file_list:
            if logfile[-len(LoggingHandler.__LOG_FILE_EXTENSION):] != LoggingHandler.__LOG_FILE_EXTENSION:
                continue
            if logfile[:len(LoggingHandler.__LOG_FILE_NAME)] != LoggingHandler.__LOG_FILE_NAME:
                continue
            number = logfile[len(LoggingHandler.__LOG_FILE_NAME):-len(LoggingHandler.__LOG_FILE_EXTENSION)]
            if not number.isnumeric():
                continue
            number = int(number)
            if number >= log_identifier:
                log_identifier = number + 1

        return f'{LoggingHandler.__LOG_FILE_NAME}{log_identifier}{LoggingHandler.__LOG_FILE_EXTENSION}'

    @staticmethod
    def __clear_file(filename: str) -> None:
        with open(filename, 'w') as _:
            pass

    @staticmethod
    def __set_log_file_location_prefix(prefix: str) -> None:
        prefix = FileHandler.ensure_dir_exists(prefix)
        LoggingHandler.__INITIALIZED = True
        LoggingHandler.__LOG_FILE_LOCATION_PREFIX = prefix

    @staticmethod
    def init_log_file(location_prefix: str) -> None:
        LoggingHandler.__set_log_file_location_prefix(location_prefix)
        LoggingHandler.__clear_file(LoggingHandler.get_log_file())

    # endregion

    # region Logging
    @staticmethod
    def print(text: str) -> None:
        if LoggingHandler.PRINT:
            print(text)

    @staticmethod
    def log(text: str) -> None:
        with open(LoggingHandler.get_log_file(), 'a') as log_file:
            log_file.write('{0!s}\n'.format(text))

    @staticmethod
    def log_and_print(text: str) -> None:
        LoggingHandler.print(text)
        LoggingHandler.log(text)

    @staticmethod
    def prepend_timestamps_to_lines(text: str) -> str:
        text = str(text).split('\n')
        current_time = str(datetime.now())
        text = ['{0}: {1}'.format(current_time, line) for line in text]
        text = '\n'.join(text)
        return text

    @staticmethod
    def log_and_print_prepend_timestamps(text: str) -> None:
        text = LoggingHandler.prepend_timestamps_to_lines(text)
        LoggingHandler.log_and_print(text)

    # endregion
