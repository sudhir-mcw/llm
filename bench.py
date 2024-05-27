import sys
import os 
from threading import Thread
import psutil


def bench(thread,file):
    cpu_measures = []
    cpu_core_measures = []
    memory_measures = []

    while thread.is_alive():
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        cpu_core_usage = psutil.cpu_percent(interval=0.1,percpu=True)


        memory_measures.append(memory_usage)
        cpu_measures.append(cpu_usage)
        cpu_core_measures.append(cpu_core_usage)


    print("mem usage : ",sum(memory_measures)/len(memory_measures))
    print("cpu usage : ",sum(cpu_measures)/len(cpu_measures))
    
    import numpy as np
    cpu_core_measures = np.array(cpu_core_measures)
    cpu_core_measures_avg = [np.mean(cpu_core_measures[:,i]) for i in range(cpu_core_measures.shape[1])]

    print("cpu core usage : ",cpu_core_measures_avg)

def run_process(file):
    path = os.path.join(os.getcwd(),file)
    os.system(f'/home/mcwaiteam/py312/bin/python {path}')


if __name__ == "__main__":
    file = "run_inference.py"

    thread1 = Thread(target=run_process,args=(file,))
    thread2 = Thread(target=bench,args=(thread1,file))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()


    print("thread1 ",thread1,thread1.is_alive())
    print("thread2 ",thread2,thread2.is_alive())