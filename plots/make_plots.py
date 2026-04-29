import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

# Set the cwd to parent
os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
MEDIA_DST = Path.cwd() / 'plots' 

if __name__ == '__main__':

  # Experiment: Varying the Buffer Size
  # 
  # 3 lines:
  # - With GPU enabled, CPU disabled
  # - With GPU disabled, CPU enabled
  # - With both GPU and CPU enabled
  buf_sz = pd.read_csv("results/buf_sz.csv")
  
  # cpu_serial = buf_sz[(buf_sz['gpu_enabled'] == 0) & (buf_sz['cpu_enabled'] == 1) & (buf_sz['exe_mode'] == 'SERIAL')]['e2e_lat_avg'].tolist()
  # cpu_dag = buf_sz[(buf_sz['gpu_enabled'] == 0) & (buf_sz['cpu_enabled'] == 1) & (buf_sz['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  # gpu_serial = buf_sz[(buf_sz['gpu_enabled'] == 1) & (buf_sz['cpu_enabled'] == 0) & (buf_sz['exe_mode'] == 'SERIAL')]['e2e_lat_avg'].tolist()
  gpu_dag = buf_sz[(buf_sz['gpu_enabled'] == 1) & (buf_sz['cpu_enabled'] == 0) & (buf_sz['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  # hybrid_dag = buf_sz[(buf_sz['gpu_enabled'] == 1) & (buf_sz['cpu_enabled'] == 1) & (buf_sz['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  bufs = buf_sz[(buf_sz['gpu_enabled'] == 1) & (buf_sz['cpu_enabled'] == 0) & (buf_sz['exe_mode'] == 'DAG')]['buf_size'].tolist()
  
  plt.figure()
  # plt.plot(buf_szs, cpu_serial, label='CPU Serial', marker='o', linestyle='-', color='blue')
  # plt.plot(buf_szs, cpu_dag, label='CPU DAG', marker='o', linestyle='-', color='orange')
  # plt.plot(buf_szs, gpu_serial, label='GPU Serial', marker='o', linestyle='-', color='green')
  plt.plot(bufs, gpu_dag, label='GPU DAG', marker='o', linestyle='-', color='red')
  # plt.plot(buf_szs, hybrid_dag, label='Hybrid DAG', marker='o', linestyle='-', color='purple')
  
  plt.title('E2E Batch Latency vs. Buffer Size')
  plt.xlabel('Buffer Size')
  plt.xscale('log', base=2)
  plt.ylabel('Latency (ms)')
  plt.legend()
  plt.grid()
  plt.savefig(MEDIA_DST / 'buf_sz.png')
  # plt.show() 
  
  matrix_sz = pd.read_csv("results/matrix_sz.csv")
  
  # cpu_serial = matrix_sz[(matrix_sz['gpu_enabled'] == 0) & (matrix_sz['cpu_enabled'] == 1) & (matrix_sz['exe_mode'] == 'SERIAL')]['e2e_lat_avg'].tolist()
  cpu_dag = matrix_sz[(matrix_sz['gpu_enabled'] == 0) & (matrix_sz['cpu_enabled'] == 1) & (matrix_sz['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  # gpu_serial = matrix_sz[(matrix_sz['gpu_enabled'] == 1) & (matrix_sz['cpu_enabled'] == 0) & (matrix_sz['exe_mode'] == 'SERIAL')]['e2e_lat_avg'].tolist()
  gpu_dag = matrix_sz[(matrix_sz['gpu_enabled'] == 1) & (matrix_sz['cpu_enabled'] == 0) & (matrix_sz['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  hybrid_dag = matrix_sz[(matrix_sz['gpu_enabled'] == 1) & (matrix_sz['cpu_enabled'] == 1) & (matrix_sz['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  bufs = matrix_sz[(matrix_sz['gpu_enabled'] == 1) & (matrix_sz['cpu_enabled'] == 0) & (matrix_sz['exe_mode'] == 'DAG')]['buf_size'].tolist()
  mat_szs = matrix_sz[(matrix_sz['gpu_enabled'] == 1) & (matrix_sz['cpu_enabled'] == 0) & (matrix_sz['exe_mode'] == 'DAG')]['mat_size'].tolist()
  
  plt.figure()
  # plt.plot(mat_szs, cpu_serial, label='CPU Serial', marker='o', linestyle='-', color='blue')
  plt.plot(mat_szs, cpu_dag, label='CPU DAG', marker='o', linestyle='-', color='orange')
  # plt.plot(mat_szs, gpu_serial, label='GPU Serial', marker='o', linestyle='-', color='green')
  plt.plot(mat_szs, gpu_dag, label='GPU DAG', marker='o', linestyle='-', color='red')
  plt.plot(mat_szs[:-2], hybrid_dag, label='Hybrid DAG', marker='o', linestyle='-', color='purple')
  
  plt.title('E2E Batch Latency vs. Matrix Size')
  plt.xlabel('Matrix Size')
  plt.xscale('log', base=2)
  plt.ylabel('Latency (ms)')
  plt.legend()
  plt.grid()
  plt.savefig(MEDIA_DST / 'matrix_sz.png')
  # plt.show() 
  
  num_state_mat = pd.read_csv("results/num_state_mat.csv")

  # cpu_serial = num_state_mat[(num_state_mat['gpu_enabled'] == 0) & (num_state_mat['cpu_enabled'] == 1) & (num_state_mat['exe_mode'] == 'SERIAL')]['e2e_lat_avg'].tolist()
  cpu_dag = num_state_mat[(num_state_mat['gpu_enabled'] == 0) & (num_state_mat['cpu_enabled'] == 1) & (num_state_mat['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  # gpu_serial = num_state_mat[(num_state_mat['gpu_enabled'] == 1) & (num_state_mat['cpu_enabled'] == 0) & (num_state_mat['exe_mode'] == 'SERIAL')]['e2e_lat_avg'].tolist()
  gpu_dag = num_state_mat[(num_state_mat['gpu_enabled'] == 1) & (num_state_mat['cpu_enabled'] == 0) & (num_state_mat['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  hybrid_dag = num_state_mat[(num_state_mat['gpu_enabled'] == 1) & (num_state_mat['cpu_enabled'] == 1) & (num_state_mat['exe_mode'] == 'DAG')]['e2e_lat_avg'].tolist()
  bufs = num_state_mat[(num_state_mat['gpu_enabled'] == 1) & (num_state_mat['cpu_enabled'] == 0) & (num_state_mat['exe_mode'] == 'DAG')]['buf_size'].tolist()
  num_state_matrices = num_state_mat[(num_state_mat['gpu_enabled'] == 1) & (num_state_mat['cpu_enabled'] == 0) & (num_state_mat['exe_mode'] == 'DAG')]['num_state_mat'].tolist()
  
  plt.figure()
  # plt.plot(num_state_matrices, cpu_serial, label='CPU Serial', marker='o', linestyle='-', color='blue')
  plt.plot(num_state_matrices, cpu_dag, label='CPU DAG', marker='o', linestyle='-', color='orange')
  # plt.plot(num_state_matrices, gpu_serial, label='GPU Serial', marker='o', linestyle='-', color='green')
  plt.plot(num_state_matrices, gpu_dag, label='GPU DAG', marker='o', linestyle='-', color='red')
  plt.plot(num_state_matrices, hybrid_dag, label='Hybrid DAG', marker='o', linestyle='-', color='purple')
  
  plt.title('E2E Batch Latency vs. Number of State Matrices')
  plt.xlabel('Number of State Matrices')
  plt.xscale('log', base=2)
  plt.ylabel('Latency (ms)')
  plt.legend()
  plt.grid()
  plt.savefig(MEDIA_DST / 'num_state_mat.png')
  # plt.show() 