#pragma once

#include <vector>

namespace cpu {

struct CriticalPoints {
  std::vector<int> maxes;
  std::vector<int> saddles;
  std::vector<int> mins;
};

}  // namespace cpu
