#include <map>
#include <set>
#include <vector>
#include <fstream>

#include "workload.h"

// enum class OpType : uint8_t {
//   SCALAR_ADD = 0,
//   SCALAR_SUB = 1,
//   SCALAR_MULT = 2,
//   MAT_ADD = 3,
//   MAT_SUB = 4,
//   MAT_MULT = 5,
//   NEW_MAT_ADD = 6,
//   NEW_MAT_SUB = 7,
//   NEW_MAT_MULT = 8
// };

struct DagNode {
    SerializedOp op; 
    std::vector<uint8_t> mat_data; 
    std::set<uint64_t> deps; 
    bool has_fused_scalar; 
    int fused_scalar; 
}; //end dagnode struct

class DagGenerator {
private: 
    std::map<uint64_t, uint64_t> last_write; 
    std::map<uint64_t, DagNode> dag; 

    bool heavy_op(OpType type) {
        return type == OpType::MAT_MULT || type == OpType::MAT_ADD || type == OpType::MAT_SUB || type == OpType::NEW_MAT_MULT 
            || type == OpType::NEW_MAT_ADD || type == OpType::NEW_MAT_SUB;
    } //end heavy operation 

public: 
    void build_dag(const std::string& path) {
        std::ifstream log(path, std::ios::binary);
        SerializedOp sop; 

        while (log.read(reinterpret_cast<char*>(&sop), sizeof(SerializedOp))) {
            uint64_t targ_mat = sop.dest_mat_id_1;

            if (last_write.count(targ_mat)) {
                uint64_t prev_op = last_write[targ_mat]; 
                DagNode& prev = dag[prev_op]; 

                //merge scalar ops 
                if (sop.type == prev.op.type && (sop.type == OpType::SCALAR_ADD || sop.type == OpType::SCALAR_SUB || sop.type == OpType::SCALAR_MULT) && last_write.count(targ_mat)) {
                    // [KAP325] I've assumed that when we do scalar subtrctions numbers are passed in as positves 
                    if (prev.op.type == OpType::SCALAR_ADD || prev.op.type == OpType::SCALAR_SUB) {
                        prev.op.scalar_param += sop.scalar_param; 
                    } //end if 

                    if (prev.op.type == OpType::SCALAR_MULT) {
                        prev.op.scalar_param *= sop.scalar_param; 
                    } //end if 

                    last_write[sop.id] = prev_op; 
                    continue; 
                } //end if 

                //kernel fuxzion 
                if ((sop.type == OpType::SCALAR_ADD || sop.type == OpType::SCALAR_MULT) && heavy_op(prev.op.type)) {
                    uint64_t prev_op = last_write[targ_mat]; 
                    DagNode& prev = dag[prev_op]; 

                    int val = (sop.type == OpType::SCALAR_SUB) ? -sop.scalar_param : sop.scalar_param;
                    prev.fused_scalar += val; 
                    prev.has_fused_scalar = true; 
                    last_write[sop.id] = prev_op; 
                    continue; 
                } //end if 
            } //end if 

            DagNode node; 
            node.op = sop; 
            node.has_fused_scalar = false;
            node.fused_scalar = 0;

            if (last_write.count(sop.dest_mat_id_1)) {
                node.deps.insert(last_write[sop.dest_mat_id_1]); 
            } //end if 

            if (sop.type == OpType::MAT_ADD || sop.type == OpType::MAT_SUB || sop.type == OpType::MAT_MULT) {
                if (last_write.count(sop.dest_mat_id_2)) {
                    node.deps.insert(last_write[sop.dest_mat_id_2]); 
                } //end if 
            } //end if 

            last_write[sop.dest_mat_id_1] = sop.id; 

            if (sop.has_mat_param) {
                uint64_t r, c;
                log.read(reinterpret_cast<char*>(&r), sizeof(uint64_t));
                log.read(reinterpret_cast<char*>(&c), sizeof(uint64_t));
                node.mat_data.resize(r * c * sizeof(int));
                log.read(reinterpret_cast<char*>(node.mat_data.data()), node.mat_data.size());
            } //end iof 

            dag[sop.id] = node; 
        } //end while 
    } //end bukld dag 

    const std::map<uint64_t, DagNode>& get_dag() const {
        return dag; 
    } //end dag getter 
}; //end class