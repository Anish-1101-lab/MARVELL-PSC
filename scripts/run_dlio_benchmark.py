import sys
import builtins
import time
import os

original_open = builtins.open

trace_headers_written = set()

def write_trace_safe(workload, ts, pid, tid, name, cat, dur):
    trace_dir = f"./real_traces/{workload}"
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = f"{trace_dir}/dlio_trace.csv"
    
    if workload not in trace_headers_written:
        if not os.path.exists(trace_path):
            with original_open(trace_path, "wb", buffering=0) as f:
                f.write(b"ts,pid,tid,name,cat,dur\n")
        trace_headers_written.add(workload)
        
    line = f"{ts},{pid},{tid},{name},{cat},{dur}\n".encode()
    with original_open(trace_path, "ab", buffering=0) as f:
        f.write(line)

def hooked_open(file, mode='r', *args, **kwargs):
    start = time.time()
    try:
        res = original_open(file, mode, *args, **kwargs)
    except Exception as e:
        raise e
    dur = time.time() - start
    
    if isinstance(file, str) and "data/" in file and (".npz" in file or ".tfrecord" in file or ".jpeg" in file):
        workload = "resnet" if "resnet" in file else "bert" if "bert" in file else "unet3d"
        name = os.path.basename(file)
        write_trace_safe(workload, start, os.getpid(), 0, name, "read", dur)
        
    return res

builtins.open = hooked_open

from dlio_benchmark.main import main
if __name__ == '__main__':
    main()
