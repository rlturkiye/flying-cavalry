import os
import signal
import subprocess
import time

TOTAL_MINUTES_RESET = 3600*3
TOTAL_TIME_RESET = TOTAL_MINUTES_RESET*60

# path checkpoints/
def find_new_name(path):
    #find last modified folder (to find which checkpoint folder) 2xtimes
    og_path = os.getcwd()
    os.chdir(path)
    all_subdirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    # file name should be path+filename to check isdir.
    path = os.path.join(path,latest_subdir)
    os.chdir(latest_subdir) 
    all_subdirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    path = os.path.join(path,latest_subdir)
    os.chdir(latest_subdir) 
    all_subdirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    path = os.path.join(path,latest_subdir)
    os.chdir(latest_subdir)
    # dqnn131513 -- checkpoint0015 -
    #inside folder find biggest file name
    files = os.listdir(".")
    largest_file_size = 0
    largest_file = ""
    for file in files:
        if os.path.isfile(file):
            size = os.path.getsize(file)
            if size >  largest_file_size:
                largest_file_size = size
                largest_file = file
    path=os.path.join(path,largest_file)
    os.chdir(og_path)
    print(og_path+path) 
    return os.path.join(og_path,path)

CHECKPOINT_PATH = "checkpoints"
def loop():
    while(True):
        time.sleep(35)
        proc = subprocess.Popen('../LinuxNoEditor/RLTurkiyeVersion1.sh > /dev/null 2>&1', shell=True,preexec_fn=os.setsid)
        time.sleep(35)
        check = find_new_name(CHECKPOINT_PATH)
        proc2 = subprocess.Popen('python3 rllib/main.py --restore_path '+ check, shell=True,preexec_fn=os.setsid)
        time.sleep(TOTAL_TIME_RESET)
        os.killpg(proc.pid, signal.SIGTERM)
        os.killpg(proc2.pid, signal.SIGTERM)
loop()

