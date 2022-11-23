import json, os, sys
import datetime as dt
import pickle

class Logger:
    def __init__(self, log_dir='runs/', log_name=None):
        if log_dir is None:
            self.log_dir = None
            return

        log_base_dir = log_dir
        if log_base_dir:
            os.makedirs(log_dir, exist_ok=True)
        if log_name is None:
            log_name = f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.log_dir = os.path.join(log_base_dir, log_name)
        os.makedirs(self.log_dir)
        
        self.log_file_path = os.path.join(self.log_dir, 'run.log')
        self.log_file = open(self.log_file_path, 'w')

        print(f"Logging to directory {self.log_dir}")

    def write(self, data):
        if self.log_dir is None: return
        self.log_file.write(json.dumps(data) + '\n')
    
    def dump(self, data, name):
        if self.log_dir is None: return
        fpath = os.path.join(self.log_dir, name + '.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)