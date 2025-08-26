import fcntl, os, contextlib, time

@contextlib.contextmanager
def gpu_lock(path: str, timeout: float = 60.0, poll: float = 0.2):
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    start = time.time()
    acquired = False
    try:
        while time.time() - start < timeout:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(poll)
        if not acquired:
            raise TimeoutError("GPU lock timeout")
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
