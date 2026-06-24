#pragma once

#include <torch/extension.h>

struct DMSComplex {
  torch::Tensor shape;

  // Critical Points & Edges
  torch::Tensor max_pts;
  torch::Tensor min_pts;
  torch::Tensor sad_pts;
  torch::Tensor e_max;
  torch::Tensor e_min;
  torch::Tensor p_max;
  torch::Tensor p_min;
  torch::Tensor ppairs_max;
  torch::Tensor ppairs_min;
  torch::Tensor grad_indices;

  // Maxima Attraction (Ascending)
  torch::Tensor peaks;
  torch::Tensor ridges;
  torch::Tensor ridge_offsets;
  torch::Tensor ridge_arc_offsets;

  // Minima Attraction (Descending)
  torch::Tensor basins;
  torch::Tensor valleys;
  torch::Tensor valley_offsets;
  torch::Tensor valley_arc_offsets;
};