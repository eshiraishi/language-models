def litstr(chars):
    return "".join(repr(char).replace("'", "") for char in chars)
