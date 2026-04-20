#pragma once

#include <vector>
#include <map>
#include <queue>
#include "dag.h" 

class Scheduler {
public:
    static std::vector<std::vector<uint64_t>> get_levels(const std::map<uint64_t, DagNode>& dag) {
        std::vector<std::vector<uint64_t>> lvls;
        std::map<uint64_t, int> in_degree;
        std::map<uint64_t, std::vector<uint64_t>> children;

        for (auto const& [id, node] : dag) {
            in_degree[id] = node.deps.size();
            for (uint64_t parent_id : node.deps) {
                children[parent_id].push_back(id);
            } //end for
        } //end for 

        std::vector<uint64_t> curr_lvl;
        for (auto const& [id, node] : dag) {

            //nodes with no deps (in degree is 0)
            if (in_degree[id] == 0) {
                curr_lvl.push_back(id);
            } //end if 
        } //end for 

        while (!curr_lvl.empty()) {
            lvls.push_back(curr_lvl);
            std::vector<uint64_t> next_level;

            for (uint64_t node_id : curr_lvl) {
                //for ezch children of nodes in current lvls
                for (uint64_t child_id : children[node_id]) {
                    in_degree[child_id]--;

                    if (in_degree[child_id] == 0) { //all deps satisfied, add to next lvl 
                        next_level.push_back(child_id);
                    } //end if 
                } //end for 
            } //end for 
            curr_lvl = next_level;
        } //end while 

        return lvls;
    } //end get_lvsl 

    // [KAP325] print scheudle for debug purposes (we don't actually need this for production )
    static void print(const std::vector<std::vector<uint64_t>>& lvls) {
        for (size_t i = 0; i < lvls.size(); ++i) {
            printf("Level %zu: ", i);

            for (uint64_t id : lvls[i]) {
                printf("%lu ", id);
            } //endf or 
            
            printf("\n");
        } //end for 
    } //end print
}; //end scheduler class