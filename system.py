#print data about available GPUs, CPUs, RAM, etc.
import torch
import psutil
import os

print("Available GPUs: ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" - GPU ", i, " memory: ", torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024, "GB")

#cpus aviailable on this node
print("Available CPUs: ", psutil.cpu_count(logical=False))

print("Available RAM: ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
print("Total RAM: ", psutil.virtual_memory().total / 1024 / 1024 / 1024, "GB")