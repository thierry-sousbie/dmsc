#pragma once

#include <vector>

#include "../cpu/arcs_topology_struct.hxx"
#include "../managed_tensor.hxx"
#include "./gradient_struct.hxx"

namespace gpu {
struct WSHelpers {
  ManagedTensor temp_flat_max;
  ManagedTensor temp_flat_min;

  WSHelpers() : temp_flat_max("temp_flat_max", false), temp_flat_min("temp_flat_min", false) {}

  void reset() {}
};
struct Workspace {
  int H;
  int W;
  int Nx;
  int num_cells;

  WSHelpers hlp;
  GradientData gradient_data;
  SaddleNodes saddle_nodes;

  Workspace(int H, int W) : H(H), W(W), Nx(W + 1), num_cells(4 * (H + 1) * (W + 1)) {}

  void reset() {
    hlp.reset();
  }
};
}  // namespace gpu
