#pragma once

#include <torch/extension.h>

struct CellGroupsData {
  torch::Tensor vertex_groups;  // size = {H, W}
  torch::Tensor face_groups;    // size = {H+1, W+1}
};