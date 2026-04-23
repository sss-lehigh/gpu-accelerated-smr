#include <iostream>
#include <string>

//dag builder, scheduler, executor 
#include "dag.h"
#include "scheduler.h"
#include "gpu.cu"

const uint64_t ROWS = 10000;
const uint64_t COLS = 10000;
const std::string LOG_PATH = "dummy_smr.log";

int main(int argc, char* argv[]) {
    try {
        std::cout << "[1/4] Initializing DAG Generator..." << std::endl;
        DagGenerator builder;

        builder.build_dag(LOG_PATH);
        auto& dag = builder.get_dag();
        std::cout << "Done. Nodes in DAG: " << dag.size() << std::endl;

        std::cout << "[2/4] Sorting DAG into parallel levels..." << std::endl;
        auto levels = Scheduler::get_levels(dag);
        
        Scheduler::print(levels); // [KAP325] for debug/maybe we can add it to our slides 

        std::cout << "[3/4] Allocating VRAM and creating streams..." << std::endl;
        GpuExecutor executor(ROWS, COLS);
        executor.load_state(init_state); 

        std::cout << "[4/4] Executing on GPU..." << std::endl;
        executor.run(dag, levels);

        std::cout << "SUCCESS: Workload completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } //end try catch 

    return EXIT_SUCCESS;
} //end main 