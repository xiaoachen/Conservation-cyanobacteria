import numpy as np
import pandas as pd
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys  # 关键修改：导入 sys 模块
import subprocess
import time
import json
import threading
import signal

def get_input_json(file_path, name, seq):
    json_data = {
        "name": name,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": seq
                }
            }
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }
    json_filename = f"{name}.json"
    json_path = os.path.join(file_path, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

def process_chunk(chunk, num_worker, record, gpu_id, af3_root, concurrency, cpu_per_task, stop_event):
    output_dir_base = f"{dir}/output/output_{num_worker}"
    input_dir_local = f"{dir}/input/input_{num_worker}"
    
    python_script_path = os.path.join(af3_root, "run_alphafold.py")
    
    tsv_file = f"{dir}/val_protein_record_{num_worker}.tsv"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tasks = []
    for _, row in chunk.iterrows():
        protein = row["aa"]
        id_name = row["WP_accession"]
        get_input_json(input_dir_local, id_name, protein)
        tasks.append((id_name, protein, os.path.join(input_dir_local, f"{id_name}.json")))

    active = []
    idx = 0
    try:
        while (idx < len(tasks) or active) and not stop_event.is_set():
            while idx < len(tasks) and len(active) < concurrency and not stop_event.is_set():
                id_name, protein_seq, input_json_path = tasks[idx]
                idx += 1

                try:
                    output_path = Path(os.path.join(output_dir_base, f"{id_name}"))
                    if output_path.exists():
                        print(f"{output_path} 已存在，跳过")
                        continue
                    print(f"{output_path} 不存在，开始生成 (GPU: {gpu_id})")

                    if len(protein_seq) > 3000:
                        print(f"序列 {id_name} 长度大于3000，不支持")
                        record.loc[record.shape[0]] = [id_name, protein_seq]
                        record.to_csv(tsv_file, sep=",", index=False)
                        continue

                    args = [
                        "--json_path", str(input_json_path),
                        "--model_dir", os.path.join(af3_root, "params/"),
                        "--output_dir", str(output_path),
                        "--jax_compilation_cache_dir", os.path.join(af3_root, "jax_dir"),
                        "--jackhmmer_n_cpu", str(cpu_per_task),
                        "--nhmmer_n_cpu", str(cpu_per_task)
                    ]

                    process = subprocess.Popen(
                        [sys.executable, python_script_path] + args,
                        cwd=af3_root,
                        env=env
                    )
                    active.append((process, id_name, protein_seq))

                except Exception as e:
                    print(f"!!! [GPU {gpu_id} | {id_name}] 发生异常: {e}")
                    record.loc[record.shape[0]] = [id_name, protein_seq]
                    record.to_csv(tsv_file, sep=",", index=False)
                    print(f"已记录失败样本到 record_{num_worker}.tsv")

            i = 0
            while i < len(active):
                proc, id_name, protein_seq = active[i]
                ret = proc.poll()
                if ret is None:
                    i += 1
                    continue
                if ret != 0:
                    print(f"!!! [GPU {gpu_id} | {id_name}] 运行失败，返回码：{ret}")
                    record.loc[record.shape[0]] = [id_name, protein_seq]
                    record.to_csv(tsv_file, sep=",", index=False)
                else:
                    print(f"[GPU {gpu_id} | {id_name}] 完成")
                active.pop(i)
            
            # Short sleep to avoid busy loop
            if active:
                time.sleep(1)
                
    finally:
        # Terminate any running processes if we exit (e.g. due to stop_event)
        for proc, id_name, _ in active:
            if proc.poll() is None:
                print(f"[GPU {gpu_id}] Terminating process for {id_name}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

def process_chunks_with_threads(chunks, records, gpu_ids, af3_root, concurrency, cpu_per_task):
    stop_event = threading.Event()
    futures = []
    
    # Handle SIGINT to set stop_event
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    
    def signal_handler(sig, frame):
        print("\nReceived KeyboardInterrupt, stopping workers...")
        stop_event.set()
        # Restore original handler so subsequent Ctrl+C works normally (force kill)
        signal.signal(signal.SIGINT, original_sigint_handler)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            for worker_idx, (chunk, gpu_id) in enumerate(zip(chunks, gpu_ids)):
                print(f"启动线程 {worker_idx}，使用 GPU {gpu_id}")
                futures.append(executor.submit(process_chunk, chunk, worker_idx, records[worker_idx], gpu_id, af3_root, concurrency, cpu_per_task, stop_event))
            
            # Wait for all futures
            # We don't need to explicitly wait here as context manager does it, 
            # but we want to monitor stop_event or handle exceptions?
            # Actually context manager __exit__ will wait.
            pass
            
    except KeyboardInterrupt:
        print("\nMain thread interrupted. Stopping...")
        stop_event.set()
        # Allow time for threads to cleanup
    finally:
        stop_event.set()
        # Restore signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", default="achen/test1", type=str, help="Path to the main data directory")
    argparser.add_argument("--gpus", default="0,1,2,3", type=str, help="Comma-separated list of GPU IDs to use")
    
    argparser.add_argument("--af3_root", default="/home/liangce/alphafold3", type=str, help="Path to the AlphaFold 3 root directory")
    argparser.add_argument("--concurrency_per_gpu", default=1, type=int, help="Number of concurrent jobs per GPU")
    argparser.add_argument("--cpu_per_task", default=4, type=int, help="Number of CPUs per Jackhmmer task")
    
    args = argparser.parse_args()
    
    gpu_ids = list(map(int, args.gpus.split(",")))
    max_numworkers = len(gpu_ids)
    print(f"使用的GPU列表: {gpu_ids}")
    print(f"最大工作线程数: {max_numworkers}")
    
    dir = args.input_dir
    af3_root_path = args.af3_root
    concurrency = max(1, int(args.concurrency_per_gpu))
    cpu_per_task = args.cpu_per_task

    df = pd.read_csv(f"{dir}/seq.csv", encoding="utf-8")
    # df['protein'] = df['protein'].apply(lambda x: x[:-1] if x.endswith('*') else x)
    df['aa'] = df['aa'].apply(lambda x: x[:-1] if x.endswith('*') else x)
    
    if not os.path.exists(f"{dir}/output"):
        os.makedirs(f"{dir}/output")
    if not os.path.exists(f"{dir}/input"):
        os.makedirs(f"{dir}/input")
        
    records = []
    for i in range(max_numworkers):
        if not os.path.exists(f"{dir}/output/output_{i}"):
            os.makedirs(f"{dir}/output/output_{i}")
        if not os.path.exists(f"{dir}/input/input_{i}"):
            os.makedirs(f"{dir}/input/input_{i}")
        tsv_file = f"{dir}/val_protein_record_{i}.tsv"
        if os.path.exists(tsv_file):
            print(f"找到已存在的记录文件: {tsv_file}")
            record = pd.read_csv(tsv_file)
        else:
            print(f"未找到记录文件，创建新的: {tsv_file}")
            record = pd.DataFrame(columns=['id', 'sequence'])
        records.append(record)

    chunks = np.array_split(df, max_numworkers)
    process_chunks_with_threads(chunks, records, gpu_ids, af3_root_path, concurrency, cpu_per_task)



# import numpy as np
# import pandas as pd
# import os
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from pathlib import Path
# import sys
# import logging
# import json
# import subprocess
# import argparse
# # 定义一个处理每个子集的函数
# # def process_chunk(chunk, num_worker, record):

# # 获取当前脚本文件的绝对路径
# # script_dir = os.path.dirname(os.path.abspath(__file__))
# # # 切换工作目录到脚本所在目录
# # os.chdir(script_dir)

# # print("当前工作目录已切换到脚本目录:", os.getcwd())

# def get_input_json(file_path, name, seq):
#     # Generate JSON files for the last two sequences
#     json_data = {
#         "name": name,
#         "sequences": [
#             {
#                 "protein": {
#                     "id": ["A"],
#                     "sequence": seq
#                 }
#             }
#         ],
#         "modelSeeds": [1],
#         "dialect": "alphafold3",
#         "version": 1
#     }
#     # Save JSON file
#     json_filename = f"{name}.json"
#     json_path = os.path.join(file_path, json_filename)
#     with open(json_path, 'w') as json_file:
#         json.dump(json_data, json_file, indent=2)



# def process_chunk(chunk, num_worker, record, gpu_id):
#     # 这里可以添加对每个子集的处理逻辑
#     # num_worker
#     output_dir_base = f"{dir}/output/output_{num_worker}"
#     input_dir = f"{dir}/input/input_{num_worker}"
#     python_file = "run_alphafold.py"
#     # 为每个工作线程定义正确的tsv文件路径
#     tsv_file = f"{dir}/val_protein_record_{num_worker}.tsv"
#     # # 设置环境变量
#     # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     # 设置环境变量 - 使用传入的GPU ID

#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     for i, row in chunk.iterrows():
#         protein = row["protein"]
#         id = row["name"]
#         # print(protein)
#         # print(id)
#         get_input_json(input_dir, id, protein)


#     for file in os.listdir(input_dir):

#         # print(file[:-5]) # QCD40802.1|GH43_16.json
#         id = file[:-5]
#         protein = chunk.loc[chunk["name"] == id, "protein"].values[0]
#         # print(protein)
#         print(os.path.join(input_dir, file))
#         input_json = os.path.join(input_dir, file)
#         try:
#             output_dir = Path(os.path.join(output_dir_base, f"{id}"))
#             if os.path.exists(output_dir):
#                 print(f"{output_dir}已存在")
#                 continue
#             print(f"{output_dir}不存在，开始生成")

#             if len(protein) > 3000:
#                 print("不支持长度大于3000的氨基酸序列")
#                 record.loc[record.shape[0]] = [id, protein]
#                 record.to_csv(tsv_file, sep=",", index=False)
#                 continue

#             # 要传递的参数
#             args = ["--json_path", input_json, 
#                     "--model_dir", "params/",
#                     "--output_dir", output_dir,
#                     "--jax_compilation_cache_dir", "jax_dir"
#                     ]
#             s = subprocess.Popen(
#                 [f"python", python_file] + args,  # 调用命令
#                 env=env,  # 关键：传递自定义环境变量
#                 # check=True,  # 如果命令返回非零退出码，抛出异常
#                 # text=True,  # 将输出解码为字符串
#                 # capture_output=True  # 捕获标准输出和标准错误
#             )
#             s.wait()
#         except subprocess.CalledProcessError as e:
#             # 如果命令执行失败
#             print("命令执行失败，返回码：", e.returncode)
#             print("错误输出：", e.stderr)
#             record.loc[record.shape[0]] = [id, protein]
#             record.to_csv(tsv_file, sep=",", index=False)
#             print(f"保存到record_{num_worker}文件")
#         except Exception as e:
#             print(f"GPU {gpu_id} 启动失败: {e}")
#             record.loc[record.shape[0]] = [id, protein]
#             record.to_csv(tsv_file, sep=",", index=False)
#             print(f"保存到record_{num_worker}文件")


# # 使用多线程处理每个子集
# def process_chunks_with_threads(chunks, records, gpu_ids):
#     with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
#         for worker_idx, (chunk, gpu_id) in enumerate(zip(chunks, gpu_ids)):
#             print(f"启动线程 {worker_idx}，使用 GPU {gpu_id}")
#             executor.submit(process_chunk, chunk, worker_idx, records[worker_idx], gpu_id)



# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     argparser.add_argument("--input_dir", default="liang/test1", type=str, help="Path to the parsed PDBs")
#     argparser.add_argument("--gpus", default="1,2,3,4", type=str, help="Path to the parsed PDBs")
#     args = argparser.parse_args()
#     gpu_ids = list(map(int, args.gpus.split(",")))  # 按逗号分割
#     # 指定要使用的GPU ID
#     # gpu_ids = [0, 1, 2, 3]
#     max_numworkers = len(gpu_ids)
#     print(gpu_ids)
#     print(max_numworkers)
#     dir = args.input_dir

#     df = pd.read_csv(f"{dir}/seq.csv", encoding="utf-8")
#     # 判断最后一个字符是否为 '*'，并去掉 '*'
#     df['protein'] = df['protein'].apply(lambda x: x[:-1] if x.endswith('*') else x)
#     # print(df)
#     if not os.path.exists(f"{dir}/output"):
#         os.makedirs(f"{dir}/output")
#     if not os.path.exists(f"{dir}/input"):
#         os.makedirs(f"{dir}/input")
#     records = []
#     for i in range(max_numworkers):  #创建文件夹，以保存子线程数据
#         if not os.path.exists(f"{dir}/output/output_{i}"):
#             os.makedirs(f"{dir}/output/output_{i}")
#         if not os.path.exists(f"{dir}/input/input_{i}"):
#             os.makedirs(f"{dir}/input/input_{i}")
#         tsv_file = f"{dir}/val_protein_record_{i}.tsv"
#         if os.path.exists(tsv_file):
#             print("存在tsv文件")
#             record = pd.read_csv(tsv_file)
#         else:
#             print("不存在tsv文件")
#             record = pd.DataFrame(columns=['id', 'sequence'])
#             # 这里好像不对吧

#         records.append(record)
#     print(record)
#     # 均等分割
#     chunks = np.array_split(df, max_numworkers)
#     process_chunks_with_threads(chunks, records, gpu_ids)

