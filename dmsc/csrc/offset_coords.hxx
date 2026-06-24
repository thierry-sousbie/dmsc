#pragma once
#include <torch/extension.h>

// This is to offset the coordinates of crtiical points for visualisation ...
inline torch::Tensor offset_coords(const torch::Tensor& pts) {
  if (!pts.defined() || pts.size(0) == 0) return torch::empty({0, 2}, torch::kFloat32);
  int N = pts.size(0);
  auto out = torch::empty({N, 2}, torch::kFloat32);
  auto pts_acc = pts.accessor<int32_t, 2>();
  auto out_acc = out.accessor<float, 2>();
  for (int i = 0; i < N; ++i) {
    float y = pts_acc[i][0];
    float x = pts_acc[i][1];
    int c_type = pts_acc[i][2];
    if (c_type == 2 || c_type == 3) y += 0.5f;
    if (c_type == 1 || c_type == 3) x += 0.5f;
    out_acc[i][0] = y;
    out_acc[i][1] = x;
  }
  return out;
}