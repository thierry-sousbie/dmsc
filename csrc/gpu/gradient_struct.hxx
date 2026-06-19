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
  ManagedTensor d_paired_with;
  ManagedTensor d_saddles;

  GradientData() : d_paired_with("d_paired_with", false), d_saddles("d_saddles", false) {}

  void reset() {}
};
}  // namespace gpu
