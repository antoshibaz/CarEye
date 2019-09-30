import sys


def print_one_line(text):
    sys.stdout.write("\r\x1b[K" + text.__str__())
