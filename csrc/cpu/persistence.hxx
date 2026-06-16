#pragma once

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "../union_find.hxx"
#include "./arcs_geometry_struct.hxx"
#include "./arcs_topology_struct.hxx"
#include "./cell_compare.hxx"

template <bool IS_DUAL>
void compute_ppairs_and_simplify(float persistence_threshold, bool trace_arcs, std::vector<int>& fast_crit_map,
                                 std::vector<SadEvent>& sorted_min_saddles, std::vector<SadEvent>& sorted_max_saddles,
                                 std::vector<int>& crit_mins, std::vector<int>& crit_maxes,
                                 const std::vector<float>& crit_min_vals, const std::vector<float>& crit_maxes_vals,
                                 std::vector<bool>& min_alive, std::vector<bool>& max_alive, UnionFind& uf_min,
                                 UnionFind& uf_max, std::vector<CancelEvent>& min_cancellations,
                                 std::vector<CancelEvent>& max_cancellations) {
  RECORD_FUNCTION("persistence_cpu", {});
  std::vector<int> safe_crit_map;
  if (trace_arcs) safe_crit_map = fast_crit_map;

  tbb::parallel_invoke(
      // Minima
      [&]() {
        UnionFind uf_min_global = uf_min;

        for (auto& ev : sorted_min_saddles) {
          int r1_g = uf_min_global.find(ev.c1_mid);
          int r2_g = uf_min_global.find(ev.c2_mid);

          if (r1_g == r2_g) {
            ev.pair_id = -1;
            ev.persistence = -1.0f;
            continue;  // Cycles
          }

          int survivor_g = -1, dying_g = -1;

          if (r1_g == -1) {
            dying_g = r2_g;
          } else if (r2_g == -1) {
            dying_g = r1_g;
          } else {
            float val1 = crit_min_vals[r1_g];
            float val2 = crit_min_vals[r2_g];
            if (value_greater<IS_DUAL>(val1, r1_g, val2, r2_g)) {
              survivor_g = r2_g;
              dying_g = r1_g;
            } else {
              survivor_g = r1_g;
              dying_g = r2_g;
            }
          }

          ev.persistence = IS_DUAL ? crit_min_vals[dying_g] - ev.s_val : ev.s_val - crit_min_vals[dying_g];
          ev.pair_id = crit_mins[dying_g];

          uf_min_global.unite_from_root(dying_g, survivor_g);

          if (ev.persistence <= persistence_threshold) {
            fast_crit_map[ev.saddle_id] = -1;
            min_alive[dying_g] = false;

            int dying = uf_min.find(ev.c1_mid);
            int survivor = uf_min.find(ev.c2_mid);
            if (dying_g == r2_g) std::swap(dying, survivor);
            uf_min.unite_from_root(dying, survivor);

            if (trace_arcs) {
              int s_idx = safe_crit_map[ev.saddle_id];
              min_cancellations.push_back(
                  {ev.saddle_id, s_idx, crit_mins[dying], dying, ev.persistence, /*is_max=*/false});
            }
          }
        }
      },

      // Maxima
      [&]() {
        UnionFind uf_max_global = uf_max;

        for (auto& ev : sorted_max_saddles) {
          int r1_g = uf_max_global.find(ev.c1_mid);
          int r2_g = uf_max_global.find(ev.c2_mid);

          if (r1_g == r2_g) {
            ev.pair_id = -1;
            ev.persistence = -1.0f;
            continue;  // Cycles
          }

          int survivor_g = -1, dying_g = -1;

          if (r1_g == -1) {
            dying_g = r2_g;
          } else if (r2_g == -1) {
            dying_g = r1_g;
          } else {
            float val1 = crit_maxes_vals[r1_g];
            float val2 = crit_maxes_vals[r2_g];
            if (value_less<IS_DUAL>(val1, r1_g, val2, r2_g)) {
              survivor_g = r2_g;
              dying_g = r1_g;
            } else {
              survivor_g = r1_g;
              dying_g = r2_g;
            }
          }

          ev.persistence = IS_DUAL ? ev.s_val - crit_maxes_vals[dying_g] : crit_maxes_vals[dying_g] - ev.s_val;
          ev.pair_id = crit_maxes[dying_g];

          uf_max_global.unite_from_root(dying_g, survivor_g);

          if (ev.persistence <= persistence_threshold) {
            fast_crit_map[ev.saddle_id] = -1;
            max_alive[dying_g] = false;

            int dying = uf_max.find(ev.c1_mid);
            int survivor = uf_max.find(ev.c2_mid);
            if (dying_g == r2_g) std::swap(dying, survivor);
            uf_max.unite_from_root(dying, survivor);

            if (trace_arcs) {
              int s_idx = safe_crit_map[ev.saddle_id];
              max_cancellations.push_back(
                  {ev.saddle_id, s_idx, crit_maxes[dying], dying, ev.persistence, /*is_max=*/true});
            }
          }
        }
      });

  {
    RECORD_FUNCTION("erase_saddles", {});
    tbb::parallel_invoke(
        [&]() {
          sorted_min_saddles.erase(std::remove_if(sorted_min_saddles.begin(), sorted_min_saddles.end(),
                                                  [&](const auto& e) { return fast_crit_map[e.saddle_id] < 0; }),
                                   sorted_min_saddles.end());
        },
        [&]() {
          sorted_max_saddles.erase(std::remove_if(sorted_max_saddles.begin(), sorted_max_saddles.end(),
                                                  [&](const auto& e) { return fast_crit_map[e.saddle_id] < 0; }),
                                   sorted_max_saddles.end());
        });
  }
}