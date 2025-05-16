import datetime


class Logger:

    def __init__(self, path, exp_name):
        self.file = open('{}/{}.log'.format(path, exp_name), 'w')

    def log(self, content, isprint=True):
        if isprint:
            print(content)
        self.file.write(str(content) + '\n')
        self.file.flush()
