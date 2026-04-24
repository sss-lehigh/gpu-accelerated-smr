// For benchmarking

#pragma once

#include "dag.h"
#include "gpu.cu"
#include "scheduler.h"
#include "cpu.cpp"

#define ARG_COLS 'c'
#define ARG_HELP 'h'
#define ARG_MATS 'm'
#define ARG_NOPS 'o'
#define ARG_PATH 'p'
#define ARG_ROWS 'r'
#define ARG_SIZE 's'



uint64_t ROWS     = 10000;
uint64_t COLS     = 10000;
uint64_t SIZE     =     0;
uint64_t num_ops  =  1000;
uint64_t num_mats =     5;

std::string LOG_PATH = "dummy_smr.log";

int main(int argc, char** argv) {

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

    cpu_executor cpu(ROWS, COLS);

    

    try {

        std::cout << "Executing on CPU..."
        /// TODO: replace with vector of ops from consensus
        cpu.run_seq(LOG_PATH);

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

    std::cout << endl;

    //////////////////////////////////////////////////////////////////////////////////////

    std::cout << "CPU parallel (using DAG)" << std::endl << std::endl;

    DagGenerator builder;

    try {

        std::cout << "[1/3] Initializing DAG Generator..." << std::endl;

        /// TODO: replace with vector of ops from consensus
        builder.build_dag(LOG_PATH);
        auto& dag = builder.get_dag();
        std::cout << "Done. Nodes in DAG: " << dag.size() << std::endl;

        std::cout << "[2/3] Sorting DAG into parallel levels..." << std::endl;
        auto levels = Scheduler::get_levels(dag);
        

        std::cout << "[3/3] Executing on CPU..." << std::endl;
        cpu.run(dag, levels);

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
        executor.run_seq(LOG_PATH);

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

     std::cout << endl;
    
    //////////////////////////////////////////////////////////////////////////////////////

    std::cout << "GPU parallel (using DAG)" << std::endl << std::endl;

    try {
        std::cout << "[1/3] Rebuilding DAG..." << std::endl;

        /// TODO: replace with vector of ops from consensus
        builder.build_dag(LOG_PATH);
        auto& dag = builder.get_dag();
        std::cout << "Done. Nodes in DAG: " << dag.size() << std::endl;

        std::cout << "[2/3] Sorting DAG into parallel levels..." << std::endl;
        auto levels = Scheduler::get_levels(dag);

        std::cout << "[3/3] Executing on GPU..." << std::endl;
        executor.run(dag, levels);

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

    return EXIT_SUCCESS;
} //end main 