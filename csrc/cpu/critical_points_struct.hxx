#pragma once

#include <vector>

struct CriticalPoints {
  std::vector<int> maxes;
  std::vector<int> saddles;
  std::vector<int> mins;

  void reset() {
    maxes.clear();
    saddles.clear();
    mins.clear();
  }
};