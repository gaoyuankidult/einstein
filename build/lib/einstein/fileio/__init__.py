from einstein.tools import save_pickle_file
from einstein.tools import load_pickle_file




def pattern_extraction_from_file(read_filename="", write_filename="", iter_dict = None, *args):
    """
    extract patterns from file and save lines that contains that pattern in to another file

    :param read_filename:  file to be processed(analyzed)
    :param write_filename:  file used to save information in read file
    :param additional iterator of objects to be write after one pattern is matched. It should be of a dictionary type.
     e.g. {["check ":"1", "2","4"]} means put 1, 2, 4 iteratively after check is found
    :param args: patterns need to be extracted
    :return:
    """
    with open(read_filename, 'r') as f_r, open(write_filename, 'w') as f_w:
        line = f_r.readline()
        counter = 0
        if iter_dict is not None:
            iter_keys = iter_dict.keys()
        args_counter = {}
        for arg in args:
            args_counter[arg] = 0
        while line:
            for arg in args:
                if arg in line:
                    f_w.write(line)
                    args_counter[arg] += 1
                    if iter_dict is not None and arg in iter_keys:
                        f_w.write(str(iter_dict[arg].next()) + "\n")
                    break
            line = f_r.readline()
            counter += 1
            if counter % 1000 == 0:
                print("%d lines have been processed" % counter)
        for arg in args:
            print("found \"%s\" %d times." % (arg, args_counter[arg]))


class FileIO(object):

    def __init__(self):
        super(FileIO, self).__init__()

    def openfile(self, filename):
        self.filename = filename
        self.f = open(self.filename, 'w')

    def save_line(self, value):
        self.f.write(str(value) + '\n')

    def closefile(self):
        try:
            self.f.close()
        except AttributeError:
            print("AttributeError: probably you didn't open file first. ")

    def save_pickle(self, data, filename):
        save_pickle_file(data, filename)

    def load_pickle(self, filename):
        return load_pickle_file(filename)
