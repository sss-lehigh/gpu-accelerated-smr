#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <unordered_map>

#include "kernels/matrix_ops.h" 
#include "scheduler.h"
#include "kernels/common.cuh"
#include "DenseMat.h"
#include "state.h"
#include "dag.h"

class GpuExecutor {
private:
    float *device_mats[5];      // Ptrs to 5 state mats
    float *stream_workspace[8]; // Pre allocated workspace for MAT_MULT
    cudaStream_t streams[8];    // Concurrent hardware queues
    uint64_t rows, cols;

    // Map to store CUDA events for dependency tracking
    std::unordered_map<uint64_t, cudaEvent_t> node_events;

    // Map to store pre-allocated VRAM pointers for node parameters
    std::unordered_map<uint64_t, float*> d_op_params;

public:
    GpuExecutor(uint64_t r, uint64_t c) : rows(r), cols(c) {
        for (int i = 0; i < 5; i++) {
            // allocate memory for 5 state mats 
            CUDA_CHECK(cudaMalloc(&device_mats[i], rows * cols * sizeof(float)));
        } //end for

        // Initialize streams and workspaces
        for (int i = 0; i < 8; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&stream_workspace[i], rows * cols * sizeof(float)));
        }
    } //end constructor

    ~GpuExecutor() {
        for (int i = 0; i < 5; ++i) {
            cudaFree(device_mats[i]);
        }

        for (int i = 0; i < 8; ++i) {
            cudaFree(stream_workspace[i]);
            cudaStreamDestroy(streams[i]);
        }

        for (auto& pair : node_events) {
            cudaEventDestroy(pair.second);
        }

        // Clean up the parameter matrices stored in VRAM
        for (auto& pair : d_op_params) {
            cudaFree(pair.second);
        }
    } // end destructor

    void load_state(const State<float>& initial_state) {
        size_t bytes = rows * cols * sizeof(float); 

        for (int i = 0; i < 5; i++) {
            const DenseMat<float>& cpu_mat = initial_state.getMatrix(i);
            CUDA_CHECK(cudaMemcpy(device_mats[i], cpu_mat.data(), bytes, cudaMemcpyHostToDevice));
        } //end for 

        CUDA_CHECK(cudaDeviceSynchronize());
    } //end load state

    // Preparation Phase
    void prepare_dag(const std::map<uint64_t, DagNode>& dag) {
        for (const auto& pair : dag) {
            const DagNode& node = pair.second;
            
            // If the node has host data, allocate VRAM and transfer it now
            if (node.h_mat_param != nullptr) {
                float* d_ptr;
                size_t bytes = node.rows * node.cols * sizeof(float);
                
                CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
                CUDA_CHECK(cudaMemcpy(d_ptr, node.h_mat_param, bytes, cudaMemcpyHostToDevice));
                
                d_op_params[node.operation.id] = d_ptr; 
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run(const std::map<uint64_t, DagNode>& dag, std::vector<std::vector<uint64_t>>& levels) {
        // Pre create events for every node in the DAG
        for (const auto& pair : dag) {
            cudaEventCreateWithFlags(&node_events[pair.first], cudaEventDisableTiming);
        }

        int s_idx = 0;

        for (const auto& level : levels) {
            for (uint64_t op_id : level) {
                const DagNode& node = dag.at(op_id);
                int current_stream_idx = s_idx % 8;
                cudaStream_t stream = streams[current_stream_idx];

                // Asynnchronous dependency resolution
                // tell this stream to wait until the streams handling the dependencies
                // have recored their completion event
                for (uint64_t dep_id : node.deps) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream, node_events[dep_id], 0));
                }

                // Launch the kernel on the specific stream
                launch(node, stream, current_stream_idx);

                // Record that this node has finished its work in this stream
                CUDA_CHECK(cudaEventRecord(node_events[op_id], stream));

                s_idx++;
            }
        }

        // Only synchronize ONCE at the very end of the entire DAG execution
        CUDA_CHECK(cudaDeviceSynchronize());
    } //end run 

private:
    void launch(const DagNode& node, cudaStream_t stream, int stream_idx) {
        float* d_out = device_mats[node.operation.dest_mat_id_1.value()];

        if (node.has_fused_scalar) {
            launchFusedScalarMultiplyAndAdd(d_out, node.fused_alpha, node.fused_beta, rows, cols, stream);
        } else {
            switch (node.operation.type) {
                case OpType::SCALAR_ADD:
                    launchAddScalar(d_out, (float)node.operation.scalar_param.value(), rows, cols, stream);
                    break;

                case OpType::SCALAR_SUB:
                    launchSubtractScalar(d_out, (float)node.operation.scalar_param.value(), rows, cols, stream);
                    break;

                case OpType::SCALAR_MULT:
                    launchMultiplyScalar(d_out, (float)node.operation.scalar_param.value(), rows, cols, stream);
                    break;

                case OpType::MAT_MULT: 
                {
                    float *d_mat_B = device_mats[node.operation.dest_mat_id_2.value()];
                    float *d_temp_result = stream_workspace[stream_idx]; // use pre allocated buffer
                    size_t size = rows*cols*sizeof(float);

                    // SGEMM into temp buffer, then Async Copy back
                    launchSgemm(d_out, d_mat_B, d_temp_result, rows, cols, cols, stream);
                    CUDA_CHECK(cudaMemcpyAsync(d_out, d_temp_result, size, cudaMemcpyDeviceToDevice, stream));
                    break;
                }

                case OpType::NEW_MAT_ADD:
                {
                    float* d_param = d_op_params[node.operation.id];
                    launchInPlaceMatrixAdd(d_out, d_param, rows, cols, stream); 
                    break; 
                }

                case OpType::NEW_MAT_SUB:
                {
                    float* d_param = d_op_params[node.operation.id];
                    launchInPlaceMatrixSub(d_out, d_param, rows, cols, stream); 
                    break;
                }

                case OpType::NEW_MAT_MULT:
                {
                    float* d_param = d_op_params[node.operation.id];
                    float* d_temp_result = stream_workspace[stream_idx]; 
                    size_t size = rows * cols * sizeof(float);

                    launchSgemm(d_out, d_param, d_temp_result, rows, cols, cols, stream);
                    CUDA_CHECK(cudaMemcpyAsync(d_out, d_temp_result, size, cudaMemcpyDeviceToDevice, stream));
                    break;
                }
                case OpType::ELEMAT_MULT:
                {
                    float* d_param = d_op_params[node.operation.id];
                    launchInPlaceElementwiseMatrixMult(d_out, d_param, rows, cols, stream);
                    break;
                }

                default:
                    break;
            } //end switch 
        } //end if else
    } //end launch 
}; //end gpuexecutor class 