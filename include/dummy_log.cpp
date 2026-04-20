#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include "workload.h"

// g++ -O3 -std=c++20 dummy_log.cpp -I. -o log_gen

int main() {
    std::srand(std::time(nullptr)); 
    WorkloadGenerator generator; 

    //1000 ops with 5 matrices 
    uint64_t num_ops = 1000; 
    uint64_t num_mats = 5; 

    std::cout << "Generating " << num_ops << " operations..." << std::endl;
    generator.generate(num_ops, num_mats); 

    std::cout << "Writing to dummy_smr.log..." << std::endl;
    generator.write_log("dummy_smr.log"); 
    std::cout << "All done." << std::endl;

    return 0; 
} //end main 