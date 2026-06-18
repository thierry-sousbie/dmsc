#pragma once
#include <torch/extension.h>

#include <vector>

namespace gpu {
struct CriticalPoints {
  std::vector<int> maxes;
  std::vector<int> saddles;
  std::vector<int> mins;
};

struct GradientData {
  CriticalPoints cp;
  std::vector<int> paired_with;

  // Handles for GPU memory. keep data between calls so we don t need to reallocate
  torch::Tensor d_data;
  torch::Tensor d_paired_with;
  torch::Tensor d_saddles;
};
}  // namespace gpu
