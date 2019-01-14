from multiprocessing import Process, Queue
def f(q):
    for k in range(1):
        for i in range(10000000):
            j=10
        f=j
    print ('Hello')
    for i in range(10):
        q.put(i)
if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    item=0
    while item!=9:
        item=q.get(block=True, timeout=None)
        print (item)
    p.join()