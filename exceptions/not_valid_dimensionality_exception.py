class NotValidDimensionalityException(Exception):
    def __init__(self, variable, dimensionality):
        message = f"Invalid dimensionality of variable '{variable}', expected dimensionality {dimensionality}."
        super().__init__(message)
