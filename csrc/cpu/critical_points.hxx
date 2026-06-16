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

#include "./critical_points_struct.hxx"

void extract_critical_points(const std::vector<int>& paired_with, int H, int W, int Nx, std::vector<int>& crit_mins,
                             std::vector<int>& crit_saddles, std::vector<int>& crit_maxes) {
  RECORD_FUNCTION("extract_critical_points_cpu", {});
  size_t num_cells = paired_with.size();
  const int* paired_ptr = paired_with.data();

  // use one container per thread to avoid sync ...
  tbb::enumerable_thread_specific<CriticalPoints> local_cps;
  tbb::parallel_for(tbb::blocked_range<int>(0, num_cells, 1024), [&](const tbb::blocked_range<int>& r) {
    CriticalPoints& local_cp = local_cps.local();

    for (int id = r.begin(); id != r.end(); ++id) {
      if (paired_ptr[id] == -1) {
        int type = get_type(id);
        int y = get_y(id, Nx);
        int x = get_x(id, Nx);

        if (type == 0) {
          if (is_valid_v(y, x, H, W)) local_cp.mins.push_back(id);
        } else if (type == 1) {
          if (is_valid_ehx(y, x, H, W)) local_cp.saddles.push_back(id);
        } else if (type == 2) {
          if (is_valid_evy(y, x, H, W)) local_cp.saddles.push_back(id);
        } else if (type == 3) {
          if (is_valid_f(y, x, H, W)) local_cp.maxes.push_back(id);
        }
      }
    }
  });

  size_t total_mins = 0;
  size_t total_saddles = 0;
  size_t total_maxes = 0;

  for (const auto& local_cp : local_cps) {
    total_mins += local_cp.mins.size();
    total_saddles += local_cp.saddles.size();
    total_maxes += local_cp.maxes.size();
  }
  crit_mins.clear();
  crit_mins.reserve(total_mins);
  crit_saddles.clear();
  crit_saddles.reserve(total_saddles);
  crit_maxes.clear();
  crit_maxes.reserve(total_maxes);

  for (auto& local_cp : local_cps) {
    crit_mins.insert(crit_mins.end(), local_cp.mins.begin(), local_cp.mins.end());
    crit_saddles.insert(crit_saddles.end(), local_cp.saddles.begin(), local_cp.saddles.end());
    crit_maxes.insert(crit_maxes.end(), local_cp.maxes.begin(), local_cp.maxes.end());
  }
}
