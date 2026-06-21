#pragma once

#include <vector>

#include "./critical_points.hxx"

namespace cpu {

struct GradientData {
  CriticalPoints cp;
  std::vector<int> paired_with;

  void reset() {
    // cp.reset();
    // paired_with.clear();
  }
};

}  // namespace cpu