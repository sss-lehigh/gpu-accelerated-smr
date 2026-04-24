#pragma once
#include "scheduler.h"
#include "cpu_matrix_ops.cpp"
#include "cpu_matrix_ops_seq.cpp"
#include "workload.h"

class cpu_executor {
private:
    float* device_mats[5]; //ptrs to 5 state mats
    uint64_t rows, cols;

public:
    cpu_executor(uint64_t r, uint64_t c) : rows(r), cols(c) {
        
    } //end constructor 

    void run(const std::map<uint64_t, DagNode>& dag, std::vector<std::vector<uint64_t>>& levels) {
        for (const auto& level : levels) {
            for (uint64_t op_id : level) {
                const DagNode& node = dag.at(op_id);
                
                launch(node);
            } //end for 
        } //end for 
    } //end run 

    void run_seq(const std::vector<op>& log) {        
        launch_seq(log);
    } //end run 

private:
    void launch(const DagNode& node) {
        float* d_out = device_mats[node.op.dest_mat_id_1];

        if (node.has_fused_scalar) {
            // launchScaleAndAdd(d_out, 1.0f, (float)node.fused_scalar, rows, cols);
        } else {
            switch (node.op.type) {
                case OpType::SCALAR_ADD:
                    launchAddScalar(d_out, (float)node.op.scalar_param, rows, cols);
                    break;

                case OpType::SCALAR_SUB:
                    launchAddScalar(d_out, -(float)node.op.scalar_param, rows, cols);
                    break;

                case OpType::SCALAR_MULT:
                    launchMultiplyScalar(d_out, (float)sop.scalar_param, rows, cols);
                    break;

                case OpType::MAT_MULT: 
                {
                    float* d_temp_result;
                    float* d_mat_B = device_mats[node.op.dest_mat_id_2];

                    launchSgemm(d_out, d_mat_B, d_temp_result, rows, cols, cols);

                    break;
                }

                case OpType::NEW_MAT_ADD:
                {
                    float* d_temp_result;
                    size_t size = rows*cols*sizeof(float); 

                    launchMatrixAdd(d_out, node.d_mat_param, d_temp_result, rows, cols); 

                    break; 
                }

                case OpType::NEW_MAT_SUB:
                {
                    float* d_temp_result;

                    launchMatrixSub(d_out, node.d_mat_param, d_temp_result, rows, cols); 


                    break; 
                }

                case OpType::NEW_MAT_MULT:
                {
                    float* d_temp_result;

                    launchSgemm(d_out, node.d_mat_param, d_temp_result, rows, cols, cols);

                    break;
                }

                default:
                    break;
            } //end switch 
        } //end if else
    } //end launch 

    void launch_seq(const std::vector<op>& path) {

        // struct op {
        //     uint64_t id;
        //     OpType type;
        //     std::optional<uint64_t> dest_mat_id_1;
        //     std::optional<uint64_t> dest_mat_id_2;
        //     // Will only be populated if we have a "new" prefix, indicating that this
        //     // will have additional matrix as the parameter instead of index
        //     std::optional<int> scalar_param;
        //     std::optional<DenseMat<int>> mat_param;
        // };

        for (auto node = path->begin(); node != path->end(); node++) {
            sop = *node;

            float* d_out = device_mats[sop.dest_mat_id_1];

            switch (sop.type) {
                case OpType::SCALAR_ADD:
                    launchAddScalar_seq(d_out, (float)sop.scalar_param, rows, cols);
                    break;

                case OpType::SCALAR_SUB:
                    launchAddScalar_seq(d_out, -(float)sop.scalar_param, rows, cols);
                    break;

                case OpType::SCALAR_MULT:
                    launchMultiplyScalar_seq(d_out, (float)sop.scalar_param, rows, cols);
                    break;

                case OpType::MAT_MULT: 
                {
                    float* d_temp_result;
                    float* d_mat_B = device_mats[sop.dest_mat_id_2];

                    launchSgemm_seq(d_out, d_mat_B, d_temp_result, rows, cols, cols);


                    break;
                }

                case OpType::NEW_MAT_ADD:
                {
                    float* d_temp_result;
                    
                    float* d_mat_B = sop.mat_param;

                    launchMatrixAdd(d_out, d_mat_B, d_temp_result, rows, cols); 


                    break; 
                }

                case OpType::NEW_MAT_SUB:
                {
                    float* d_temp_result;
                    float* d_mat_B = sop.mat_param;
                    
                    launchMatrixSub_seq(d_out, d_mat_B, d_temp_result, rows, cols); 


                    break; 
                }

                case OpType::NEW_MAT_MULT:
                {
                    float* d_temp_result;
                    float* d_mat_B = sop.mat_param;

                    launchSgemm_seq(d_out, d_mat_B, d_temp_result, rows, cols, cols);


                    break;
                }

                default:
                    break;
            } //end switch 
        } //end while 

        
    } //end launch 
}; //end gpuexecutor class 