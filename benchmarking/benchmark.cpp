// For benchmarking

#pragma once

#include <chrono>

#include "cpu.cpp"
#include "dag.h"
#include "gpu.cu"
#include "scheduler.h"

// used for command line parsing
#define ARG_COLS 'c'
#define ARG_HELP 'h'
#define ARG_MATS 'm'
#define ARG_NOPS 'o'
#define ARG_PATH 'p'
#define ARG_ROWS 'r'
#define ARG_SIZE 's'



// values to be used in matrix/workload generation
uint64_t ROWS     =    2;
uint64_t COLS     =    2;
uint64_t SIZE     =    2;
uint64_t num_ops  = 1000;
uint64_t num_mats =    5;

std::string LOG_PATH = "dummy_smr.log";

int main(int argc, char** argv) {

    // Command line processing
    for (int i = 0; i < argc; i++) {
        if (argv[i][0] == "-") {
            switch (argv[i][1])
            {

            case ARG_HELP:
                std::cout << "command line arguments:" << std::endl;

                std::cout << "-" << ARG_HELP << ": display this menu"           << std::endl;

                std::cout << "-" << ARG_PATH << ": path for log file"           << std::endl;

                std::cout << "-" << ARG_MATS << ": change num. of matrices"     << std::endl;
                std::cout << "-" << ARG_NOPS << ": change num. of operations"   << std::endl;
                std::cout << "-" << ARG_SIZE << ": change size of matrices"     << std::endl;

                std::cout << "\ndefaults:" << std::endl;

                std::cout << "path: dummy_smr.log" << std::endl;
                std::cout << "num_mats: 5" << std::endl;
                std::cout << "num_ops : 1000" << std::endl;
                std::cout << "mat_size: 2" << std::endl;
                std::cout << std::endl;
                return 0;

            case ARG_PATH:
                LOG_PATH = std::to_string(argv[i + 1]);
                break;
            
            case ARG_ROWS:
                ROWS = std::atoi(argv[i + 1]);
                break;
            
            case ARG_COLS:
                COLS = std::atoi(argv[i + 1]);
                break;

            case ARG_SIZE:
                SIZE = std::atoi(argv[i + 1]);
                break;

            case ARG_NOPS:
                num_ops = std::atoi(argv[i + 1]);
                break;

            case ARG_MATS:
                num_mats = std::atoi(argv[i + 1]);
                break;

            default:
                break;
            } // end switch
        }
    } // end of command line for loop

    if (SIZE) {
        ROWS = SIZE;
        COLS = SIZE;
    }

    std::srand(std::time(nullptr)); 

    /// TODO: Maybe replace with consensus stuff?
    WorkloadGenerator generator;

    std::cout << "Generate " << num_mats << " matrices..." << std::endl;

    std::vector<DenseMat<int>> state_mats;
    state_mats.reserve(num_mats);
    for(uint64_t i = 0; i < num_mats; ++i) {
        state_mats.push_back(wg.generateMatrix(SIZE));
    }

    State<int> st(state_mats);

    std::cout << "done." << std::endl;
    

    /// TODO: replace with consensus stuff
    std::cout << "Generating " << num_ops << " operations..." << std::endl;
    generator.generate(num_ops, num_mats); 

    std::cout << "Writing to " << LOG_PATH << "..." << std::endl;
    generator.write_log(LOG_PATH); 
    std::cout << "All done." << std::endl;

    std::cout << std::endl;


    //////////////////////////////////////////////////////////////////////////////////////
   
    std::cout << "CPU sequential" << std::endl << std::endl;

    // cpu_executor is based on GpuExecutor
    cpu_executor cpu(ROWS, COLS);



    try {

        std::cout << "Executing on CPU..."
        auto cpu_seq_start = std::chrono::steady_clock::now();
        /// TODO: replace with vector of ops from consensus
        cpu.run_seq(LOG_PATH);
        auto cpu_seq_end   = std::chrono::steady_clock::now();

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

    std::cout << endl;

    //////////////////////////////////////////////////////////////////////////////////////

    std::cout << "CPU parallel (using DAG)" << std::endl << std::endl;

    DagGenerator builder;



    // This is based on the code from driver.cpp and cpu_executor is based on GpuExecutor.
    try {

        std::cout << "[1/3] Initializing DAG Generator..." << std::endl;

        /// TODO: replace with vector of ops from consensus
        builder.build_dag(LOG_PATH);
        auto& dag = builder.get_dag();
        std::cout << "Done. Nodes in DAG: " << dag.size() << std::endl;

        std::cout << "[2/3] Sorting DAG into parallel levels..." << std::endl;
        auto levels = Scheduler::get_levels(dag);
        
        std::cout << "[3/3] Executing on CPU..." << std::endl;

        auto cpu_par_start = std::chrono::steady_clock::now();
        cpu.run(dag, levels);
        auto cpu_par_end   = std::chrono::steady_clock::now();

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

     std::cout << endl;
    
    //////////////////////////////////////////////////////////////////////////////////////

    std::cout << "GPU sequential" << std::endl << std::endl;

    GpuExecutor executor(ROWS, COLS);
    


    try {

        std::cout << "[1/2] Allocating VRAM and creating streams..." << std::endl;
        


        std::cout << "[2/2] Executing on GPU..." << std::endl;
        /// TODO: replace with vector of ops from consensus
        /// TODO: @rishad write a sequential version for the GPU code because I don't know how.
        auto gpu_seq_start = std::chrono::steady_clock::now();
        executor.run_seq(LOG_PATH);
        auto gpu_seq_end   = std::chrono::steady_clock::now();

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

     std::cout << endl;
    
    //////////////////////////////////////////////////////////////////////////////////////

    std::cout << "GPU parallel (using DAG)" << std::endl << std::endl;



    // This is directly from driver.cpp (besides moving builder and executor) so it should work perfectly fine.
    try {
        std::cout << "[1/3] Rebuilding DAG..." << std::endl;

        /// TODO: replace with vector of ops from consensus
        builder.build_dag(LOG_PATH);
        auto& dag = builder.get_dag();
        std::cout << "Done. Nodes in DAG: " << dag.size() << std::endl;

        std::cout << "[2/3] Sorting DAG into parallel levels..." << std::endl;
        auto levels = Scheduler::get_levels(dag);

        std::cout << "[3/3] Executing on GPU..." << std::endl;
        auto gpu_par_start = std::chrono::steady_clock::now();
        executor.run(dag, levels);
        auto gpu_par_end   = std::chrono::steady_clock::now();

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

    std::cout << std::endl;

    std::chrono::duration<double> cpu_seq = cpu_seq_end - cpu_seq_start;
    std::chrono::duration<double> cpu_par = cpu_par_end - cpu_par_start;
    std::chrono::duration<double> gpu_seq = gpu_seq_end - gpu_seq_start;
    std::chrono::duration<double> gpu_par = gpu_par_end - gpu_par_start;

    std::cout << "cpu seq. time: " << std::chrono::milliseconds(cpu_seq).count() << " ms" << std::endl;
    std::cout << "cpu par. time: " << std::chrono::milliseconds(cpu_par).count() << " ms" << std::endl;
    std::cout << "gpu seq. time: " << std::chrono::milliseconds(gpu_seq).count() << " ms" << std::endl;
    std::cout << "gpu par. time: " << std::chrono::milliseconds(gpu_par).count() << " ms" << std::endl;

    std::cout << std::endl;

    return EXIT_SUCCESS;
} //end main 