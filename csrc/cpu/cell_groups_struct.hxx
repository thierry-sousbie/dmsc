#pragma once

#include <torch/extension.h>

#include "../managed_tensor.hxx"

namespace cpu {

struct CellGroupsData {
  ManagedTensor vertex_groups;  // size = {H, W}
  ManagedTensor face_groups;    // size = {H+1, W+1}

  void reset() {}
};

}  // namespace cpu