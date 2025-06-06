# import sys
# def error_message_detail(error,error_detail:sys):
#     #error_detail is sys, and from it, we call .exc_info() to get details about the error traceback.
#     '''error_detail.exc_info() returns 3 values:
# exc_type: type of the exception (e.g., ValueError)
# exc_value: the error instance
# exc_tb: the traceback object (gives info like which file and line caused the error)
# We're only interested in the traceback (exc_tb), so we discard the first two using _.'''
#     _,_,exc_tb=error_detail.exc_info()  #sys.exc_info()
#     file_name=exc_tb.tb_frame.f_code.co_filename
#     error_message="Error occured python script name[{0}] line number[{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
#     return error_message

# class CustomException(Exception):
#     #self is an object
#     def __init__(self,error_message,error_detail:sys):
#        #super() is a built-in function that lets you call a method from the parent (or base) class
#        super().__init__(error_message)
#        self.error_message=error_message_detail(error_message,error_detail=error_detail)
#     def __str__(self):
#         return self.error_message

import sys

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()  #sys.exc_info()

    file_name = "<unknown_file>"
    line_no = "<unknown_line>"

    if exc_tb is not None:
        file_name=exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno

    error_message="Error occured python script name[{0}] line number[{1}] error message [{2}]".format(file_name,line_no,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
       super().__init__(error_message) # Pass the string message here
       self.error_message=error_message_detail(error_message,error_detail=error_detail)
    def __str__(self):
        return self.error_message