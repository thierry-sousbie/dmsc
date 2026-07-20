#pragma once

#include <tuple>
#include <vector>

namespace cpu {

struct SadEvent {
  int saddle_id;
  int c1_mid;  // mapped id (to the MS complex datastructure)
  int c2_mid;  // mapped id (to the MS complex datastructure)
  float s_val;
  int pair_id = -1;  // cell id of the persistence pair
  float persistence = 0.0f;
};

// this is equivalent std::less if IS_DUAL is false but it becomes
// std::more in the dual space
template <bool IS_DUAL = false>
struct SadEventLess {
  inline bool operator()(const SadEvent& a, const SadEvent& b) const {
    if constexpr (!IS_DUAL) {
      return std::tie(a.s_val, a.saddle_id) < std::tie(b.s_val, b.saddle_id);
    } else {
      return std::tie(a.s_val, a.saddle_id) > std::tie(b.s_val, b.saddle_id);
    }
  }
};

template <bool IS_DUAL = false>
using SadEventGreater = SadEventLess<!IS_DUAL>;

struct ArcsTopology {
  std::vector<int> max_arcs_len;  // unsorted
  std::vector<int> min_arcs_len;  // unsorted
  std::vector<SadEvent> sorted_max_saddles;
  std::vector<SadEvent> sorted_min_saddles;
};

}  // namespace cpu
