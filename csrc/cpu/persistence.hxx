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

namespace cpu {

template <bool IS_DUAL, typename Workspace>
void compute_ppairs_and_simplify(Workspace& ws, float persistence_threshold, bool trace_max_arcs, bool trace_min_arcs) {
  RECORD_FUNCTION("persistence_cpu", {});
  auto& fast_crit_map = ws.hlp.fast_crit_map;
  auto& sorted_min_saddles = ws.arcs_topology.sorted_min_saddles;
  auto& sorted_max_saddles = ws.arcs_topology.sorted_max_saddles;
  auto& crit_mins = ws.gradient_data.cp.mins;
  auto& crit_maxes = ws.gradient_data.cp.maxes;
  auto& crit_min_vals = ws.hlp.crit_min_vals;
  auto& crit_max_vals = ws.hlp.crit_max_vals;

  auto& min_alive = ws.p_data.min_alive;
  auto& max_alive = ws.p_data.max_alive;
  auto& uf_min = ws.p_data.uf_min;
  auto& uf_max = ws.p_data.uf_max;
  auto& min_cancellations = ws.p_data.min_cancellations;
  auto& max_cancellations = ws.p_data.max_cancellations;
  auto& safe_crit_map = ws.hlp.safe_crit_map;

  min_alive.assign(crit_mins.size(), true);
  max_alive.assign(crit_maxes.size(), true);
  uf_min.reset(crit_mins.size());
  uf_max.reset(crit_maxes.size());
  if (trace_max_arcs || trace_min_arcs) {
    min_cancellations.clear();
    max_cancellations.clear();
    safe_crit_map.assign(fast_crit_map.begin(), fast_crit_map.end());
  }

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
            int cell_id1 = crit_mins[r1_g];
            int cell_id2 = crit_mins[r2_g];
            if (value_greater<IS_DUAL>(val1, cell_id1, val2, cell_id2)) {
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

            if (trace_min_arcs) {
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
            float val1 = crit_max_vals[r1_g];
            float val2 = crit_max_vals[r2_g];
            int cell_id1 = crit_maxes[r1_g];
            int cell_id2 = crit_maxes[r2_g];
            if (value_less<IS_DUAL>(val1, cell_id1, val2, cell_id2)) {
              survivor_g = r2_g;
              dying_g = r1_g;
            } else {
              survivor_g = r1_g;
              dying_g = r2_g;
            }
          }

          ev.persistence = IS_DUAL ? ev.s_val - crit_max_vals[dying_g] : crit_max_vals[dying_g] - ev.s_val;
          ev.pair_id = crit_maxes[dying_g];

          uf_max_global.unite_from_root(dying_g, survivor_g);

          if (ev.persistence <= persistence_threshold) {
            fast_crit_map[ev.saddle_id] = -1;
            max_alive[dying_g] = false;

            int dying = uf_max.find(ev.c1_mid);
            int survivor = uf_max.find(ev.c2_mid);
            if (dying_g == r2_g) std::swap(dying, survivor);
            uf_max.unite_from_root(dying, survivor);

            if (trace_max_arcs) {
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

}  // namespace cpu