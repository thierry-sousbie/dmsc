#pragma once

#include <vector>

#include "../union_find.hxx"

namespace cpu {

struct CancelEvent {
  int saddle_id;  // Dense cell ID of the saddle
  int s_idx;      // saddle index in the saddle list
  int target_id;  // Dense cell ID of the extremum target
  int t_idx;      // target index in the extremum list
  float persistence;
  bool is_max;  // True if this cancels a maximum, False if minimum

  bool operator<(const CancelEvent& other) const {
    if (persistence != other.persistence) return persistence < other.persistence;
    if (saddle_id != other.saddle_id) return saddle_id < other.saddle_id;
    return target_id < other.target_id;
  }
};

struct PersistenceData {
  UnionFind uf_max;
  UnionFind uf_min;
  std::vector<bool> max_alive;
  std::vector<bool> min_alive;
  std::vector<CancelEvent> min_cancellations;
  std::vector<CancelEvent> max_cancellations;

  void reset() {
    // max_alive.clear();
    // min_alive.clear();
    // min_cancellations.clear();
    // max_cancellations.clear();
  }
};

}  // namespace cpu