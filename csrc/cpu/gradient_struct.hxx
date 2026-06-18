#pragma once

#include <vector>

#include "./critical_points.hxx"

struct GradientData {
  CriticalPoints cp;
  std::vector<int> paired_with;

  void reset() {
    cp.reset();
    paired_with.clear();
  }
};
