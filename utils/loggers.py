import sys
import os
import os.path as osp
import time

class Logger(object):
    """Writes console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os
       >>> import os.path as osp
       >>> from utils.loggers import Logger
       >>> sys.stdout = Logger()
       >>> save_dir = sys.stdout.get_save_dir()
       >>> print('all content will be printed into record.log')
    """  
    def __init__(self):
        self.console = sys.stdout
        self.file = None
        
        dir_name = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) #_%S
        self.save_dir = osp.join('logs', dir_name)
        
        self.mkdir_if_missing(self.save_dir)
        self.file = open(osp.join(self.save_dir, 'record.log'), 'w')
                    
    def get_save_dir(self):
        return self.save_dir
            
    def mkdir_if_missing(self, dirname):
        """Creates dirname if it is missing."""
        if not osp.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()