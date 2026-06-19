#pragma once

#include <vector>

namespace cpu {

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

}  // namespace cpu