class NotValidLengthException(Exception):
    def __init__(self, variable, expected):
        message = f"Invalid length of variable '{variable}', expected {expected}."
        super().__init__(message)
