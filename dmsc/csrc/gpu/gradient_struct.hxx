#pragma once
#include <torch/extension.h>

#include <vector>

#include "../managed_tensor.hxx"

namespace gpu {
struct CriticalPoints {
  std::vector<int> maxes;
  std::vector<int> saddles;
  std::vector<int> mins;
};

struct CriticalPointsAsTensors {
  torch::Tensor maxes;
  torch::Tensor saddles;
  torch::Tensor mins;
};

struct GradientData {
  CriticalPoints cp;
  std::vector<int> paired_with;

  // Handles for GPU memory. keep data between calls so we don t need to reallocate
  ManagedTensor d_paired_with;
  ManagedTensor d_saddles;
  ManagedTensor d_maxes;
  ManagedTensor d_mins;

  GradientData()
      : d_paired_with("d_paired_with", false),
        d_saddles("d_saddles", false),
        d_maxes("d_maxes", false),
        d_mins("d_mins", false) {}
};
}  // namespace gpu
