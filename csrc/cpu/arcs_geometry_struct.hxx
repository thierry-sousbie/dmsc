#pragma once

#include <torch/extension.h>

#include <vector>

#include "../managed_tensor.hxx"

struct Arc {
  int target;  // index of the connected max/min
  int offset;  // starting index in the flat geometry array
  int length;  // number of cells forming the manifold
};

// Store arcs incident to a saddle point
struct SaddleNode {
  int alive;  // int not bool for alignment
  Arc max_arcs[2];
  Arc min_arcs[2];
};

struct SaddleNodes {
  std::vector<SaddleNode> nodes;
  ManagedTensor flat_max_geom;
  ManagedTensor flat_min_geom;

  void reset() {
    nodes.clear();
  }
};

// Priority Queue Event for Physical Simplification
// struct CancelEvent {
//   int saddle_id;  // Dense cell ID of the saddle
//   int s_idx;      // saddle index in the saddle list
//   int target_id;  // Dense cell ID of the extremum target
//   int t_idx;      // target index in the extremum list
//   float persistence;
//   bool is_max;  // True if this cancels a maximum, False if minimum

//   bool operator<(const CancelEvent& other) const {
//     if (persistence != other.persistence) return persistence < other.persistence;
//     if (saddle_id != other.saddle_id) return saddle_id < other.saddle_id;
//     return target_id < other.target_id;
//   }
// };