# import multiprocessing
# def spawn():
#   print('test!')

# if __name__ == '__main__':
#   for i in range(5):
#     p = multiprocessing.Process(target=spawn)
#     p.start()


# from multiprocessing import Process
# import os

# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())

# def f(name):
#     info('function f')
#     print('hello', name)

# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()


import multiprocessing as mp
import logging
import socket
import time

logger = mp.log_to_stderr(logging.DEBUG)

def worker(socket):
    while True:
        client, address = socket.accept()
        logger.debug("{u} connected".format(u=address))
        client.send("OK")
        client.close()
if __name__ == '__main__':
    num_workers = 5

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('',9090))
    serversocket.listen(5)

    workers = [mp.Process(target=worker, args=(serversocket,)) for i in
            range(num_workers)]

    for p in workers:
        p.daemon = True
        p.start()

    while True:
        try:
            time.sleep(10)
        except:
            break