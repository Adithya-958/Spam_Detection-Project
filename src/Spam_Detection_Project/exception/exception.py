import sys
from src.Spam_Detection_Project.logger.logger import logging
class CustomException(Exception):
    def __init__(self, message, errors:sys):
        super().__init__(message)
        self.message = message
        _,_,exc_tb = errors.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
    def __str__(self):
        return f'CustomException: Script_name is: {self.args[0]}, Error_Message is :[{1}] , Errors: {self.errors}'.format(
            self.file_name, self.line_number, str(self.message))