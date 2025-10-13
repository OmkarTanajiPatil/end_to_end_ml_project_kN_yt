import sys
import logger
import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = (
        error_detail.exc_info()
    )  # exc_tb will store the information realted to file, which error, where it occurs and all
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "\n -> File name: [{0}] \n -> Line number: [{1}] \n -> Error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero exception")
        raise CustomException(e, sys)
