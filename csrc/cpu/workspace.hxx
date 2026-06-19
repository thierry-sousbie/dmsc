#pragma once

#include <vector>

#include "../dmsc_struct.hxx"
#include "../managed_tensor.hxx"
#include "./arcs_geometry_struct.hxx"
#include "./arcs_topology_struct.hxx"
#include "./cell_groups_struct.hxx"
#include "./gradient_struct.hxx"
#include "./persistence_struct.hxx"

namespace cpu {
struct WSHelpers {
  std::vector<float> crit_min_vals;
  std::vector<float> crit_max_vals;
  // look up table of size num_cells mapping critical cell_id to index in gradient_data.cp
  std::vector<int> fast_crit_map;
  std::vector<int> safe_crit_map;

  ManagedTensor temp_flat_max;
  ManagedTensor temp_flat_min;

  WSHelpers() : temp_flat_max("temp_flat_max", false), temp_flat_min("temp_flat_min", false) {}

  void reset() {
    fast_crit_map.clear();
    safe_crit_map.clear();
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
  CellGroupsData cell_groups;
  PersistenceData p_data;

  Workspace(int H, int W) : H(H), W(W), Nx(W + 1), num_cells(4 * (H + 1) * (W + 1)) {}

  void reset() {
    gradient_data.reset();
    arcs_topology.reset();
    saddle_nodes.reset();
    cell_groups.reset();
    p_data.reset();
    hlp.reset();
  }

  // DMSComplex ouput() {
  // RECORD_FUNCTION("populate_ridges_and_valleys_cpu", {});

  // auto flat_max_geom = saddle_nodes.flat_max_geom.get();
  // auto flat_min_geom = saddle_nodes.flat_min_geom.get();

  // const int* flat_max_ptr = flat_max_geom.template data_ptr<int>();
  // const int* flat_min_ptr = flat_min_geom.template data_ptr<int>();

  // for (const auto& ev : arcs_topology.sorted_min_saddles) {
  //   int s_idx = hlp.fast_crit_map[ev.saddle_id];
  //   const auto& node = saddle_nodes.nodes[s_idx];

  //   if (!node.alive) continue;  // Safety check

  //   const int* max_arc0_start = flat_max_ptr + node.max_arcs[0].offset;
  //   ridge_faces.insert(ridge_faces.end(), max_arc0_start, max_arc0_start + node.max_arcs[0].length);
  //   arc_faces_offsets.push_back(ridge_faces.size());

  //   const int* max_arc1_start = flat_max_ptr + node.max_arcs[1].offset;
  //   ridge_faces.insert(ridge_faces.end(), max_arc1_start, max_arc1_start + node.max_arcs[1].length);
  //   ridge_faces_offsets.push_back(ridge_faces.size());

  //   const int* min_arc0_start = flat_min_ptr + node.min_arcs[0].offset;
  //   ridge_vertices.insert(ridge_vertices.end(), min_arc0_start, min_arc0_start + node.min_arcs[0].length);
  //   arc_vertices_offsets.push_back(ridge_vertices.size());

  //   const int* min_arc1_start = flat_min_ptr + node.min_arcs[1].offset;
  //   ridge_vertices.insert(ridge_vertices.end(), min_arc1_start, min_arc1_start + node.min_arcs[1].length);
  //   ridge_vertices_offsets.push_back(ridge_vertices.size());
  // }
  // }
};
}  // namespace cpu