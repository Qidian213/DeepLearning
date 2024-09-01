import os 

class suppress_stdout_stderr(object):
    def __init__(self, mode: int = 2):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(mode)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = tuple([os.dup(i+1) for i in range(mode)])
        self.mode = mode

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        for i in range(self.mode):
            os.dup2(self.null_fds[i], i+1)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        for i in range(self.mode):
            os.dup2(self.save_fds[i], i+1)
        
        for i in range(self.mode):
            os.close(self.null_fds[i])
            os.close(self.save_fds[i])