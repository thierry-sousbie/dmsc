#pragma once

#include <vector>

#include "../managed_tensor.hxx"
#include "./arcs_geometry_struct.hxx"
#include "./arcs_topology_struct.hxx"
#include "./cell_groups_struct.hxx"
#include "./gradient_struct.hxx"

namespace cpu {
struct WSHelpers {
  std::vector<float> crit_min_vals;
  std::vector<float> crit_max_vals;
  // look up table of size num_cells mapping critical cell_id to index in gradient_data.cp
  std::vector<int> fast_crit_map;

  ManagedTensor temp_flat_max;
  ManagedTensor temp_flat_min;

  WSHelpers() : temp_flat_max("temp_flat_max", false), temp_flat_min("temp_flat_min", false) {}

  void reset() {
    fast_crit_map.clear();
    crit_min_vals.clear();
    crit_max_vals.clear();
  }
};

struct Workspace {
  int H;
  int W;
  int Nx;
  int num_cells;

  WSHelpers hlp;

  GradientData gradient_data;
  ArcsTopology arcs_topology;
  SaddleNodes saddle_nodes;
  // CellGroupsData cell_groups;

  Workspace(int H, int W) : H(H), W(W), Nx(W + 1), num_cells(4 * (H + 1) * (W + 1)) {}

  void reset() {
    gradient_data.reset();
    arcs_topology.reset();
    saddle_nodes.reset();
    hlp.reset();
    // saddle_nodes.reset();
    // cell_groups.reset();
  }
};
}  // namespace cpu