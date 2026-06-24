#pragma once

#include <ATen/record_function.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <torch/extension.h>

#include <tuple>
#include <vector>

#include "../cell_complex.hxx"
#include "./arcs_topology_struct.hxx"

namespace gpu {
// Exactly as defined in metal/metal_backend.mm
struct TracedSaddlesTensors {
  torch::Tensor saddles;
  torch::Tensor max_c1;
  torch::Tensor max_c2;
  torch::Tensor min_c1;
  torch::Tensor min_c2;
  torch::Tensor s_vals;
  torch::Tensor max_len;
  torch::Tensor min_len;
};

// struct TracedSaddlesVectors {
//   std::vector<SadEvent> sorted_max_saddles;
//   std::vector<SadEvent> sorted_min_saddles;
//   std::vector<int> max_arcs_len;
//   std::vector<int> min_arcs_len;
// };

// IMPORTANT: The simulation of simplicity convention here is exactly the same as the one defined in
// SadEventLess{} / SadEventGreater{} when descending is True / False.
torch::Tensor get_packed_sort_indices(const torch::Tensor& s_vals, const torch::Tensor& saddles,
                                      bool descending = true) {
  // 1. Destroy Negative Zero
  // Adding 0.0f normalizes all -0.0 to strictly 0.0, restoring the tie-breaker.
  torch::Tensor safe_s_vals = s_vals + 0.0f;
  torch::Tensor s_bits = safe_s_vals.view(torch::kInt32);

  // Because PyTorch uses signed Int64, we must leave the top bit alone so
  // positive floats remain positive integers, and negative floats remain negative.
  // We only invert the lower 31 magnitude bits of negative floats to correct their order.
  torch::Tensor mask = s_bits < 0;
  int32_t magnitude_mask = (int32_t)0x7FFFFFFF;

  torch::Tensor s_sortable = torch::where(mask,
                                          s_bits.bitwise_xor(magnitude_mask),  // Fix negative ordering
                                          s_bits                               // Keep positives exactly as they are
  );

  torch::Tensor s_shifted = torch::bitwise_left_shift(s_sortable.to(torch::kInt64), 32);
  torch::Tensor packed_keys = torch::bitwise_or(s_shifted, saddles.to(torch::kInt64));
  return std::get<1>(torch::sort(packed_keys, -1, /*descending=*/descending)).cpu().contiguous();
}

template <bool IS_DUAL = false, typename Workspace>
inline void tensors_to_sad_events(Workspace& ws, const TracedSaddlesTensors& tensors) {
  RECORD_FUNCTION("tensors_to_sad_events", {});
  auto& out = ws.arcs_topology;
  const auto& fast_crit_map = ws.hlp.fast_crit_map;

  if (!tensors.saddles.defined() || tensors.saddles.numel() == 0) return;

  torch::Tensor indices;
  {
    RECORD_FUNCTION("sort_indices_pytorch", {});
    indices = get_packed_sort_indices(tensors.s_vals, tensors.saddles, !IS_DUAL);
  }

  // Pull everything to the CPU memory space
  torch::Tensor cpu_saddles = tensors.saddles.cpu().contiguous();
  torch::Tensor cpu_max_c1 = tensors.max_c1.cpu().contiguous();
  torch::Tensor cpu_max_c2 = tensors.max_c2.cpu().contiguous();
  torch::Tensor cpu_min_c1 = tensors.min_c1.cpu().contiguous();
  torch::Tensor cpu_min_c2 = tensors.min_c2.cpu().contiguous();
  torch::Tensor cpu_s_vals = tensors.s_vals.cpu().contiguous();
  torch::Tensor cpu_max_len = tensors.max_len.cpu().contiguous();
  torch::Tensor cpu_min_len = tensors.min_len.cpu().contiguous();

  int num_saddles = cpu_saddles.numel();

  const int* p_saddles = cpu_saddles.data_ptr<int>();
  const int* p_max_c1 = cpu_max_c1.data_ptr<int>();
  const int* p_max_c2 = cpu_max_c2.data_ptr<int>();
  const int* p_min_c1 = cpu_min_c1.data_ptr<int>();
  const int* p_min_c2 = cpu_min_c2.data_ptr<int>();
  const float* p_s_vals = cpu_s_vals.data_ptr<float>();
  const int* p_max_len = cpu_max_len.data_ptr<int>();
  const int* p_min_len = cpu_min_len.data_ptr<int>();

  // Safe pointer assignment: only active if sort_with_torch is true
  const int64_t* p_indices = indices.data_ptr<int64_t>();

  out.sorted_max_saddles.resize(num_saddles);
  out.sorted_min_saddles.resize(num_saddles);
  out.max_arcs_len.assign(num_saddles * 2, 0);
  out.min_arcs_len.assign(num_saddles * 2, 0);
  {
    RECORD_FUNCTION("CREATE_SORTED_SadEvents", {});
    tbb::parallel_for(tbb::blocked_range<int>(0, num_saddles), [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i != r.end(); ++i) {
        int max_idx = p_indices[i];

        int s_id_max = p_saddles[max_idx];
        float s_val_max = p_s_vals[max_idx];
        int max1 = p_max_c1[max_idx];
        int max2 = p_max_c2[max_idx];
        int max1_mid = (max1 != -1) ? fast_crit_map[max1] : -1;
        int max2_mid = (max2 != -1) ? fast_crit_map[max2] : -1;
        int len1 = p_max_len[2 * max_idx];
        int len2 = p_max_len[2 * max_idx + 1];

        out.sorted_max_saddles[i] = {-999, -1, -1, -1, -1, 0.0f};
        if (max1 != -1 && max2 != -1) {
          out.sorted_max_saddles[i] = {s_id_max, max1, max2, max1_mid, max2_mid, s_val_max};
          out.max_arcs_len[2 * max_idx] = len1;
          out.max_arcs_len[2 * max_idx + 1] = len2;
        } else if ((max1 != -1 && max2 == -1) || (max1 == -1 && max2 != -1)) {
          int max_real = (max1 != -1) ? max1 : max2;
          int max_mid_real = (max1 != -1) ? max1_mid : max2_mid;
          out.sorted_max_saddles[i] = {s_id_max, max_real, -1, max_mid_real, -1, s_val_max};
          out.max_arcs_len[2 * max_idx] = (max1 != -1) ? len1 : len2;
        }
      }
    });

    tbb::parallel_for(tbb::blocked_range<int>(0, num_saddles), [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i != r.end(); ++i) {
        int min_idx = p_indices[(num_saddles - 1) - i];

        int s_id_min = p_saddles[min_idx];
        float s_val_min = p_s_vals[min_idx];
        int min1 = p_min_c1[min_idx];
        int min2 = p_min_c2[min_idx];
        int min1_mid = (min1 != -1) ? fast_crit_map[min1] : -1;
        int min2_mid = (min2 != -1) ? fast_crit_map[min2] : -1;
        int len1 = p_min_len[2 * min_idx];
        int len2 = p_min_len[2 * min_idx + 1];

        out.sorted_min_saddles[i] = {-999, -1, -1, -1, -1, 0.0f};
        if (min1 != -1 && min2 != -1) {
          out.sorted_min_saddles[i] = {s_id_min, min1, min2, min1_mid, min2_mid, s_val_min};
          out.min_arcs_len[2 * min_idx] = len1;
          out.min_arcs_len[2 * min_idx + 1] = len2;
        } else if ((min1 != -1 && min2 == -1) || (min1 == -1 && min2 != -1)) {
          int min_real = (min1 != -1) ? min1 : min2;
          int min_mid_real = (min1 != -1) ? min1_mid : min2_mid;
          out.sorted_min_saddles[i] = {s_id_min, min_real, -1, min_mid_real, -1, s_val_min};
          out.min_arcs_len[2 * min_idx] = (min1 != -1) ? len1 : len2;
        }
      }
    });
  }
}
}  // namespace gpu