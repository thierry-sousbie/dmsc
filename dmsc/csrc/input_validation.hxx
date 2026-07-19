#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <limits>

inline void validate_dmsc_input(const torch::Tensor& scalar_field) {
  TORCH_CHECK(scalar_field.defined(), "Input tensor must be defined");
  TORCH_CHECK(scalar_field.layout() == c10::kStrided, "Input tensor must use strided layout");
  TORCH_CHECK(scalar_field.scalar_type() == torch::kFloat32,
              "Input tensor must have dtype torch.float32, got ", scalar_field.scalar_type());
  TORCH_CHECK(scalar_field.dim() == 2 || scalar_field.dim() == 3,
              "Input tensor must be 2D [H, W] or 3D [B, H, W], got ", scalar_field.dim(), "D");

  if (scalar_field.dim() == 3) {
    TORCH_CHECK(scalar_field.size(0) > 0, "Batch dimension must be non-empty");
  }

  const int64_t height = scalar_field.size(-2);
  const int64_t width = scalar_field.size(-1);
  TORCH_CHECK(height > 0 && width > 0, "Spatial dimensions must be non-empty, got ", height, "x", width);

  const int64_t max_cells = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(height + 1 <= max_cells / 4 / (width + 1),
              "Input is too large for 32-bit cell IDs: ", height, "x", width);
}
