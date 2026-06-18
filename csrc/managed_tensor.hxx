#pragma once
#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <string>

class ManagedTensor {
 private:
  torch::Tensor buffer;
  torch::Tensor active_tensor;  // Tracks the view from the last request
  int64_t capacity = 0;
  std::string tensor_name;
  bool debug_mode;

 public:
  explicit ManagedTensor(std::string name = "Unnamed", bool debug = false)
      : tensor_name(std::move(name)), debug_mode(debug) {}

  // Getters and setters for debugging
  const std::string& name() const {
    return tensor_name;
  }
  void set_name(const std::string& n) {
    tensor_name = n;
  }
  void set_debug(bool debug) {
    debug_mode = debug;
  }

  // Request a tensor, providing the shape and options dynamically.
  torch::Tensor request(c10::IntArrayRef shape, torch::TensorOptions options) {
    int64_t required_elements = 1;
    for (auto dim : shape) {
      required_elements *= dim;
    }

    if (required_elements == 0) {
      active_tensor = torch::empty(shape, options);
      return active_tensor;
    }

    bool needs_reallocation = false;

    // Check if we need to allocate fresh memory or grow the buffer
    if (!buffer.defined() || buffer.dtype() != options.dtype() || buffer.device() != options.device()) {
      // If uninitialized, or if the user changed the device/dtype, reset completely
      needs_reallocation = true;
      capacity = required_elements;
    } else if (required_elements > capacity) {
      // If we just need more room, grow by 1.5x to prevent thrashing
      needs_reallocation = true;
      capacity = std::max(required_elements, static_cast<int64_t>(capacity * 1.5));
    }

    if (needs_reallocation) {
      if (debug_mode) {
        std::cout << "[ManagedTensor] '" << tensor_name << "' allocating " << capacity << " elements.\n"
                  << "    Shape requested: " << shape << "\n"
                  << "    Options: " << options << "\n";
      }
      buffer = torch::empty({capacity}, options);
    }

    // Save the perfectly sized and shaped view, then return it
    active_tensor = buffer.slice(0, 0, required_elements).view(shape);
    return active_tensor;
  }

  // Get the tensor exactly as it was last requested (same shape and size)
  torch::Tensor get() const {
    if (!active_tensor.defined()) {
      std::cerr << "[ManagedTensor] Warning: get() called on '" << tensor_name << "' before it was ever requested!\n";
    }
    return active_tensor;
  }

  torch::Tensor copy_from_tensor(const torch::Tensor& src, torch::TensorOptions dest_options) {
    if (debug_mode && src.device() != dest_options.device()) {
      std::cout << "[ManagedTensor] Warning: Cross-device transfer for '" << tensor_name << "' from " << src.device()
                << " to " << dest_options.device() << ".\n";
    }
    torch::Tensor dest = request(src.sizes(), dest_options);
    dest.copy_(src);
    return dest;
  }

  // Preserve the TensorOptions from current tensor if defined, from src otherwise
  torch::Tensor copy_from_tensor(const torch::Tensor& src) {
    torch::TensorOptions target_options = buffer.defined() ? buffer.options() : src.options();
    return copy_from_tensor(src, target_options);
  }

  // Convenience method: Copies a raw CPU pointer into a properly sized device tensor
  torch::Tensor copy_from_cpu_ptr(void* data, c10::IntArrayRef shape, torch::TensorOptions options) {
    torch::TensorOptions cpu_options = options.device(torch::kCPU);
    torch::Tensor cpu_view = torch::from_blob(data, shape, cpu_options);
    torch::Tensor device_tensor = request(shape, options);

    device_tensor.copy_(cpu_view);

    return device_tensor;
  }

  // Pre-allocate memory if the size is known ahead of time
  void preallocate(c10::IntArrayRef shape, torch::TensorOptions options) {
    request(shape, options);
  }

  // Reset the buffer to reclaim memory (necessary ??)
  void clear() {
    buffer = torch::Tensor();
    active_tensor = torch::Tensor();
    capacity = 0;
  }
};