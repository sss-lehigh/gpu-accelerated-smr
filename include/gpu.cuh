#include <cuda_runtime.h>
#include "kernels/matrix_ops.h" 
#include "scheduler.h"
#include "kernels/common.cuh"
#include "DenseMat.h"
#include "state.h"

class GpuExecutor {
private:
    float* device_mats[5]; //ptrs to 5 state mats
    uint64_t rows, cols;
    cudaStream_t streams[8];

public:
    GpuExecutor(uint64_t r, uint64_t c) : rows(r), cols(c) {
        for (int i = 0; i < 5; i++) {
            //allocate memory for 5 state mats 
            CUDA_CHECK(cudaMalloc(&device_mats[i], rows * cols * sizeof(float)));
        } //end for 

        for (int i = 0; i < 8; i++) {
            cudaStreamCreate(&streams[i]);
        } //end for 
    } //end constructor 

    void load_state(const State<float>& initial_state) {
        size_t bytes = rows * cols * sizeof(float); 

        for (int i = 0; i < 5; i++) {
            const DenseMat<float>& cpu_mat = initial_state.getMatrix(i);
            CUDA_CHECK(cudaMemcpy(device_mats[i], cpu_mat.data(), bytes, cudaMemcpyHostToDevice));
        } //end for 

        CUDA_CHECK(cudaDeviceSynchronize());
    } //end load state

    void run(const std::map<uint64_t, DagNode>& dag, std::vector<std::vector<uint64_t>>& levels) {
        for (const auto& level : levels) {
            int s_idx = 0; //to allocate tasks to all 8 gpu streams 
            for (uint64_t op_id : level) {
                const DagNode& node = dag.at(op_id);
                cudaStream_t stream = streams[s_idx%8]; //determine which stream to launch on 
                
                launch(node, stream);
                s_idx++;
            } //end for 
            
            //sync for next lvl
            cudaDeviceSynchronize();
        } //end for 
    } //end run 

private:
    void launch(const DagNode& node, cudaStream_t stream) {
        float* d_out = device_mats[node.op.dest_mat_id_1];

        if (node.has_fused_scalar) {
            launchFusedScalarMultiplyAndAdd(d_out, 1.0f, (float)node.fused_scalar, rows, cols, stream);
        } else {
            switch (node.op.type) {
                case OpType::SCALAR_ADD:
                    launchAddScalar(d_out, (float)node.op.scalar_param, rows, cols, stream);
                    break;

                case OpType::SCALAR_SUB:
                    launchSubtractScalar(d_out, (float)node.op.scalar_param, rows, cols, stream);
                    break;

                case OpType::SCALAR_MULT:
                    launchMultiplyScalar(d_out, (float)node.op.scalar_param, rows, cols, stream);
                    break;

                // [KAP325] FIXME:::: we don't have cuda kernels for this
                case OpType::MAT_MULT: 
                {
                    float* d_temp_result;
                    size_t size = rows*cols*sizeof(float); 
                    cudaMalloc(&d_temp_result, size); 
                    float* d_mat_B = device_mats[node.op.dest_mat_id_2];

                    launchSgemm(d_out, d_mat_B, d_temp_result, rows, cols, cols, stream);
                    cudaMemcpyAsync(d_out, d_temp_result, size, cudaMemcpyDeviceToDevice, stream);
                    cudaFree(d_temp_result);

                    break;
                }

                case OpType::NEW_MAT_ADD:
                {
                    float* d_temp_result;
                    size_t size = rows*cols*sizeof(float); 
                    cudaMalloc(&d_temp_result, size); 
                    // float* d_mat_B = device_mats[node.op.dest_mat_id_2];

                    launchMatrixAdd(d_out, node.d_mat_param, d_temp_result, rows, cols, stream); 
                    cudaMemcpyAsync(d_out, d_temp_result, size, cudaMemcpyDeviceToDevice, stream); 
                    cudaFree(d_temp_result); 

                    break; 
                }

                case OpType::NEW_MAT_SUB:
                {
                    float* d_temp_result;
                    size_t size = rows*cols*sizeof(float); 
                    cudaMalloc(&d_temp_result, size); 
                    // float* d_mat_B = device_mats[node.op.dest_mat_id_2];

                    launchMatrixSub(d_out, node.d_mat_param, d_temp_result, rows, cols, stream); 
                    cudaMemcpyAsync(d_out, d_temp_result, size, cudaMemcpyDeviceToDevice, stream); 
                    cudaFree(d_temp_result); 

                    break; 
                }

                case OpType::NEW_MAT_MULT:
                {
                    float* d_temp_result;
                    size_t size = rows*cols*sizeof(float); 
                    cudaMalloc(&d_temp_result, size); 

                    launchSgemm(d_out,node.d_mat_param, d_temp_result, rows, cols, cols, stream);
                    cudaMemcpyAsync(d_out, d_temp_result, size, cudaMemcpyDeviceToDevice, stream);
                    cudaFree(d_temp_result);

                    break;
                }

                default:
                    break;
            } //end switch 
        } //end if else
    } //end launch 
}; //end gpuexecutor class 