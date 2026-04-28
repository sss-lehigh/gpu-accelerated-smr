#pragma once

#include <map>
#include <queue>
#include <vector>

#include "dag.h"

class Scheduler {
 public:
  static std::vector<std::vector<uint64_t>> get_levels(
      const std::map<uint64_t, DagNode>& dag) {
    std::vector<std::vector<uint64_t>> lvls;
    std::map<uint64_t, int> in_degree;
    std::map<uint64_t, std::vector<uint64_t>> children;

    for (auto const& [id, node] : dag) {
      in_degree[id] = node.deps.size();
      for (uint64_t parent_id : node.deps) {
        children[parent_id].push_back(id);
      }  // end for
    }  // end for

    std::vector<uint64_t> curr_lvl;
    for (auto const& [id, node] : dag) {
      // nodes with no deps (in degree is 0)
      if (in_degree[id] == 0) {
        curr_lvl.push_back(id);
      }  // end if
    }  // end for

    while (!curr_lvl.empty()) {
      lvls.push_back(curr_lvl);
      std::vector<uint64_t> next_level;

      for (uint64_t node_id : curr_lvl) {
        // for ezch children of nodes in current lvls
        for (uint64_t child_id : children[node_id]) {
          in_degree[child_id]--;

          if (in_degree[child_id] == 0) {  // all deps satisfied, add to next
                                           // lvl
            next_level.push_back(child_id);
          }  // end if
        }  // end for
      }  // end for
      curr_lvl = next_level;
    }  // end while

    return lvls;
  }  // end get_lvsl

  static int get_score(const std::vector<std::vector<uint64_t>>& lvls) {
    // Higher number of levels --> higher degree of conflict
    // Higher number of ops per level --> higher degree of parallelism and thus
    // more benefit from GPU The score is out of 100, and we can set a threshold
    // (e.g. 50) (0,50] runs on CPU, (50,100] runs on GPU
    if (lvls.empty()) {
      return 0;
    }
    constexpr float CPU_WEIGHT = 1.0f;
    constexpr float GPU_WEIGHT = 1.0f;
    size_t total_ops = 0;
    for (const auto& lvl : lvls) {
      total_ops += lvl.size();
    }
    const size_t num_levels = lvls.size();
    // Center the score at 50 for a "neutral" DAG 
    float score = 50.0f + static_cast<float>(total_ops) * GPU_WEIGHT -
                  static_cast<float>(num_levels) * CPU_WEIGHT;
    // Clamp to [0, 100].
    if (score < 0.0f) score = 0.0f;
    if (score > 100.0f) score = 100.0f;

    return static_cast<int>(score);
  }  // end get_score

  // [KAP325] print scheudle for debug purposes (we don't actually need this for
  // production )
  static void print(const std::vector<std::vector<uint64_t>>& lvls) {
    for (size_t i = 0; i < lvls.size(); ++i) {
      printf("Level %zu: ", i);

      for (uint64_t id : lvls[i]) {
        printf("%lu ", id);
      }  // endf or

      printf("\n");
    }  // end for
  }  // end print
};  // end scheduler class