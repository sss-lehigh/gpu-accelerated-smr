#pragma once
#include "scheduler.h"
#include "cpu_matrix_ops.cpp"

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
                    // launchScaleMatrix(d_out, (float)node.op.scalar_param, rows, cols);
                    break;

                // [KAP325] FIXME:::: we don't have cuda kernels for this
                case OpType::MAT_MULT: 
                {
                    float* d_mat_B = device_mats[node.op.dest_mat_id_2];
                }
                case OpType::NEW_MAT_ADD:
                case OpType::NEW_MAT_SUB:
                case OpType::NEW_MAT_MULT:
                    break;

                default:
                    break;
            } //end switch 
        } //end if else
    } //end launch 
}; //end gpuexecutor class 