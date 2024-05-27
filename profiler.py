import json 
import os
import sys

if len(sys.argv)==0:
    sys.argv=["","exp-logs"]


pre_process_cpu_times = []
post_process_cpu_times = []

count = 0
for file in os.listdir(os.path.join(os.getcwd(),sys.argv[1],"preprocess")):
    file = os.path.join(os.getcwd(),sys.argv[1],"preprocess",file)
    with open(file,'r') as file:
        json_data = json.load(file)
    
    trace_events = json_data["traceEvents"]
    entry_count = 0
    for event in trace_events:
        if " preprocess" in event.get("name"):
            print("preprocess each", int(event.get("dur")))
            pre_process_cpu_times.append(int(event.get("dur")))
            entry_count+=1
    if entry_count!=1:
        print("entry count : ",entry_count)
    count+=1

print("parsed files : ",count)
count   =   0
for file in os.listdir(os.path.join(os.getcwd(),sys.argv[1],"postprocess")):

    file = os.path.join(os.getcwd(),sys.argv[1],"postprocess",file)
    with open(file,'r') as file:
        json_data = json.load(file)
    
    trace_events = json_data["traceEvents"]
    entry_count = 0
    for event in trace_events:
        if "postprocess" in event.get("name"):
            print(int(event.get("dur")))
            post_process_cpu_times.append(int(event.get("dur")))
            entry_count+=1
    if entry_count!=1:
        print("entry count : ",entry_count)

    count+=1
print("parsed files : ",count)

print("pre_process_cpu_times",len(pre_process_cpu_times))

print("post_process_cpu_times",len(post_process_cpu_times))
print('sum pre time' ,sum(pre_process_cpu_times))
print('preprocess time' ,sum(pre_process_cpu_times)/len(pre_process_cpu_times))
print('postprocess time' ,sum(post_process_cpu_times)/len(post_process_cpu_times))