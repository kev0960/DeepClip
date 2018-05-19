import multiprocessing
import time
import sys

class Loader:
    def worker(self):

        print("Start doing some work..")

        while True :
            while not self.q.full() :
                time.sleep(0.2)
                self.q.put(self.total_job)
                self.total_job += 1
                print("%d job is done.. (Queue size : %d)" % (self.total_job, self.q.qsize()))

            print("Queue is full! -- leaving")

            self.cond.acquire()
            self.cond.wait()
            self.cond.release()

            print("Resume the work..")


    def __init__(self):
        self.cond = multiprocessing.Condition()
        self.p = multiprocessing.Process(target=self.worker)
        self.q = multiprocessing.Queue(maxsize=20)
        self.total_job = 0

        self.p.start()

    def consume(self):
        total = []
        for _ in range(5) :
            if not self.q.full():
                self.cond.acquire()
                self.cond.notify()
                self.cond.release()

            total.append(self.q.get())
        return total


loader = Loader()
for i in range(100):
    start_time = time.time()
    print(loader.consume())

    # training time
    time.sleep(1.2)
    elapsed_time = time.time() - start_time
    print("5 job has consumed :: ", elapsed_time)