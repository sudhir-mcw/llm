import psutil
import time
import random
from multiprocess import Pool
from statistics import mean
def my_function():
    sum = 0
    for i in range(1000000):
        sum += i
    return sum

def merge_sort(data):
    if len(data) <= 1:
        return
    
    mid = len(data) // 2
    left_data = data[:mid]
    right_data = data[mid:]
    
    merge_sort(left_data)
    merge_sort(right_data)
    
    left_index = 0
    right_index = 0
    data_index = 0
    
    while left_index < len(left_data) and right_index < len(right_data):
        if left_data[left_index] < right_data[right_index]:
            data[data_index] = left_data[left_index]
            left_index += 1
        else:
            data[data_index] = right_data[right_index]
            right_index += 1
        data_index += 1
    
    if left_index < len(left_data):
        del data[data_index:]
        data += left_data[left_index:]
    elif right_index < len(right_data):
        del data[data_index:]
        data += right_data[right_index:]
    
if __name__ == '__main__':
    # data = [11, 2, 7, 4, 5, 4, 7, 18, 9]
    data = random.sample(range(1, 100000), 90000)
    # pool.close()


    pool = Pool(8)

# Measure CPU times before the function execution
    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    print("\n cpu_times_before: ", cpu_times_before, "\n")
    cpu_freq = psutil.cpu_freq(percpu=True)
    print("cpu freq: ", cpu_freq)
    # Run the function
    # my_function()
    # merge_sort(data)
    pool.map(merge_sort, [data, data, data, data])
    
    # print("merge sort: ", data)

    # Measure CPU times after the function execution
    cpu_times_after = process.cpu_times()
    print("\n cpu_times_after: ", cpu_times_after, "\n")
    pool.close()
    # Calculate the CPU time difference
    user_time = cpu_times_after.user - cpu_times_before.user
    system_time = cpu_times_after.system - cpu_times_before.system

    # Get the CPU frequency
    cpu_freq = psutil.cpu_freq().current
    print("cpu freq: ", psutil.cpu_freq(percpu=True))
    curr_freq=[]
    for i in psutil.cpu_freq(percpu=True):
        curr_freq.append(i.current)
    print("\n mean freq: ", mean(curr_freq))
    print("\n cpu freq: ", cpu_freq)
    # Calculate CPU cycles (approximation)
    cpu_cycles_user = user_time * cpu_freq * 1e6
    cpu_cycles_system = system_time * cpu_freq * 1e6

    print(f"User CPU time: {user_time:.6f} seconds")
    print(f"System CPU time: {system_time:.6f} seconds")
    print(f"CPU cycles (user): {cpu_cycles_user:.0f}")
    print(f"CPU cycles (system): {cpu_cycles_system:.0f}")