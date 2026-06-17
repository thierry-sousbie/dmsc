#pragma once

#include "./arcs_geometry_struct.hxx"
#include "./arcs_topology_struct.hxx"
#include "./cell_groups_struct.hxx"
#include "./gradient_struct.hxx"

namespace cpu {
struct Workspace {
  GradientData gradient_data;
  ArcsTopology arcs_topology;
  SaddleNodes saddle_nodes;
  CellGroupsData cell_groups;

  void reset() {
    // gradient_data.reset();
    // arcs_topology.reset();
    // saddle_nodes.reset();
    // cell_groups.reset();
  }
};
}  // namespace cpu