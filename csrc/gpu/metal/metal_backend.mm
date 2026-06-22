#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/extension.h>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <ATen/mps/MPSAllocator.h>
#include <ATen/mps/MPSStream.h>

#include "../gradient_struct.hxx"

#include "arcs_simplification_metal.h"
#include "cell_groups_metal.h"
#include "critical_points_metal.h"
#include "gradient_metal.h"
#include "trace_from_saddles_metal.h"
#include "trace_raw_arcs_geometry_metal.h"

// Official Apple PyTorch <-> Metal Bridge
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  if (!tensor.defined()) return nil;
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

struct Constants {
  int H;
  int W;
  int Nx;
  int num_saddles;
};

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

struct MetalContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> gradient_pipeline_primal;
  id<MTLComputePipelineState> gradient_pipeline_dual;

  // New Cell Groups pipelines
  id<MTLComputePipelineState> face_groups_pipeline_primal;
  id<MTLComputePipelineState> face_groups_pipeline_dual;
  id<MTLComputePipelineState> vertex_groups_pipeline_primal;
  id<MTLComputePipelineState> vertex_groups_pipeline_dual;

  id<MTLComputePipelineState> extract_pipeline;
  id<MTLComputePipelineState> trace_pipeline_primal;
  id<MTLComputePipelineState> trace_pipeline_dual;

  id<MTLComputePipelineState> trace_raw_arcs_geometry_pipeline;

  id<MTLComputePipelineState> evaluate_simplification_pipeline;
  id<MTLComputePipelineState> contract_simplification_pipeline;
  id<MTLComputePipelineState> compress_paths_pipeline;

  bool initialized = false;
};

static MetalContext g_ctx;

inline void init_metal_context_if_needed() {
  if (g_ctx.initialized) return;

  g_ctx.device = MTLCreateSystemDefaultDevice();
  if (!g_ctx.device) throw std::runtime_error("Failed to find a suitable Metal device.");

  MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
  compileOptions.languageVersion = MTLLanguageVersion2_1;
  NSError* error = nil;

  NSString* grad_source = [NSString stringWithUTF8String:GRADIENT_METAL_SRC];
  id<MTLLibrary> grad_lib = [g_ctx.device newLibraryWithSource:grad_source options:compileOptions error:&error];
  if (!grad_lib) {
    std::string err_msg = error ? [error.localizedDescription UTF8String] : "Unknown error";
    std::cerr << "--- METAL COMPILER ERROR ---\n" << err_msg << "\n----------------------------\n";
    throw std::runtime_error("Gradient Shader Compilation Failed: " + err_msg);
  }
  g_ctx.gradient_pipeline_primal = [g_ctx.device
      newComputePipelineStateWithFunction:[grad_lib newFunctionWithName:@"compute_discrete_gradient_primal"]
                                    error:&error];
  g_ctx.gradient_pipeline_dual =
      [g_ctx.device newComputePipelineStateWithFunction:[grad_lib newFunctionWithName:@"compute_discrete_gradient_dual"]
                                                  error:&error];

  NSString* extract_source = [NSString stringWithUTF8String:CRITICAL_POINTS_METAL_SRC];
  id<MTLLibrary> extract_lib = [g_ctx.device newLibraryWithSource:extract_source options:compileOptions error:&error];
  if (!extract_lib) {
    std::string err_msg = error ? [error.localizedDescription UTF8String] : "Unknown error";
    std::cerr << "--- METAL COMPILER ERROR ---\n" << err_msg << "\n----------------------------\n";
    throw std::runtime_error("Extract Critical Points Shader Compilation Failed: " + err_msg);
  }
  g_ctx.extract_pipeline =
      [g_ctx.device newComputePipelineStateWithFunction:[extract_lib newFunctionWithName:@"extract_critical_points"]
                                                  error:&error];

  NSString* trace_source = [NSString stringWithUTF8String:TRACE_FROM_SADDLES_METAL_SRC];
  id<MTLLibrary> trace_lib = [g_ctx.device newLibraryWithSource:trace_source options:compileOptions error:&error];
  if (!trace_lib) {
    std::string err_msg = error ? [error.localizedDescription UTF8String] : "Unknown error";
    std::cerr << "--- METAL COMPILER ERROR ---\n" << err_msg << "\n----------------------------\n";
    throw std::runtime_error("trace_from_saddles Compilation Failed" + err_msg);
  }
  g_ctx.trace_pipeline_primal =
      [g_ctx.device newComputePipelineStateWithFunction:[trace_lib newFunctionWithName:@"trace_from_saddles_primal"]
                                                  error:&error];
  g_ctx.trace_pipeline_dual =
      [g_ctx.device newComputePipelineStateWithFunction:[trace_lib newFunctionWithName:@"trace_from_saddles_dual"]
                                                  error:&error];

  // Compile the new Cell Groups shader
  NSString* cell_groups_source = [NSString stringWithUTF8String:CELL_GROUPS_METAL_SRC];
  id<MTLLibrary> cell_groups_lib = [g_ctx.device newLibraryWithSource:cell_groups_source
                                                              options:compileOptions
                                                                error:&error];
  if (!cell_groups_lib) {
    std::string err_msg = error ? [error.localizedDescription UTF8String] : "Unknown error";
    std::cerr << "--- METAL COMPILER ERROR ---\n" << err_msg << "\n----------------------------\n";
    throw std::runtime_error("Cell Groups Shader Compilation Failed" + err_msg);
  }

  g_ctx.face_groups_pipeline_primal = [g_ctx.device
      newComputePipelineStateWithFunction:[cell_groups_lib newFunctionWithName:@"compute_face_groups_primal"]
                                    error:&error];
  g_ctx.face_groups_pipeline_dual = [g_ctx.device
      newComputePipelineStateWithFunction:[cell_groups_lib newFunctionWithName:@"compute_face_groups_dual"]
                                    error:&error];
  g_ctx.vertex_groups_pipeline_primal = [g_ctx.device
      newComputePipelineStateWithFunction:[cell_groups_lib newFunctionWithName:@"compute_vertex_groups_primal"]
                                    error:&error];
  g_ctx.vertex_groups_pipeline_dual = [g_ctx.device
      newComputePipelineStateWithFunction:[cell_groups_lib newFunctionWithName:@"compute_vertex_groups_dual"]
                                    error:&error];

  // Compile the Raw Arcs Geometry shader
  NSString* trace_arcs_source = [NSString stringWithUTF8String:TRACE_RAW_ARCS_GEOMETRY_METAL_SRC];
  id<MTLLibrary> trace_arcs_lib = [g_ctx.device newLibraryWithSource:trace_arcs_source
                                                             options:compileOptions
                                                               error:&error];
  if (!trace_arcs_lib) {
    std::string err_msg = error ? [error.localizedDescription UTF8String] : "Unknown error";
    std::cerr << "--- METAL COMPILER ERROR ---\n" << err_msg << "\n----------------------------\n";
    throw std::runtime_error("Trace Raw Arcs Geometry Compilation Failed: " + err_msg);
  }

  g_ctx.trace_raw_arcs_geometry_pipeline = [g_ctx.device
      newComputePipelineStateWithFunction:[trace_arcs_lib newFunctionWithName:@"trace_raw_arcs_geometry_kernel"]
                                    error:&error];

  NSString* simp_source =
      [NSString stringWithUTF8String:ARCS_SIMPLIFICATION_METAL_SRC];  // Make sure you define this string
  id<MTLLibrary> simp_lib = [g_ctx.device newLibraryWithSource:simp_source options:compileOptions error:&error];
  if (!simp_lib) {
    std::cerr << "--- METAL COMPILER ERROR ---\n" << [error.localizedDescription UTF8String] << "\n";
    throw std::runtime_error("Simplification Shader Compilation Failed");
  }
  g_ctx.evaluate_simplification_pipeline =
      [g_ctx.device newComputePipelineStateWithFunction:[simp_lib newFunctionWithName:@"evaluate_cancellations_metal"]
                                                  error:&error];
  g_ctx.contract_simplification_pipeline =
      [g_ctx.device newComputePipelineStateWithFunction:[simp_lib newFunctionWithName:@"contract_cancellations_metal"]
                                                  error:&error];
  g_ctx.compress_paths_pipeline =
      [g_ctx.device newComputePipelineStateWithFunction:[simp_lib newFunctionWithName:@"compress_paths_step_metal"]
                                                  error:&error];

  g_ctx.commandQueue = [g_ctx.device newCommandQueue];
  g_ctx.initialized = true;
}

// ==========================================
// PHASE 1: DISCRETE GRADIENT
// ==========================================
void launch_gradient_metal(torch::Tensor d_data, torch::Tensor d_paired_with, int H, int W, bool is_dual) {
  @autoreleasepool {
    init_metal_context_if_needed();

    // Ensure PyTorch has finished copying the image tensor to the GPU
    at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    id<MTLCommandBuffer> commandBuffer = [g_ctx.commandQueue commandBuffer];

    // Initialize d_paired_with to -1 natively on the GPU
    // 0xFF in every byte creates 0xFFFFFFFF, which is exactly -1 as a 32-bit int!
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder fillBuffer:getMTLBufferStorage(d_paired_with)
                      range:NSMakeRange(d_paired_with.storage_offset() * d_paired_with.itemsize(),
                                        d_paired_with.numel() * d_paired_with.itemsize())
                      value:0xFF];
    [blitEncoder endEncoding];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    if (is_dual) {
      [computeEncoder setComputePipelineState:g_ctx.gradient_pipeline_dual];
    } else {
      [computeEncoder setComputePipelineState:g_ctx.gradient_pipeline_primal];
    }

    [computeEncoder setBuffer:getMTLBufferStorage(d_data) offset:d_data.storage_offset() * d_data.itemsize() atIndex:0];
    [computeEncoder setBuffer:getMTLBufferStorage(d_paired_with)
                       offset:d_paired_with.storage_offset() * d_paired_with.itemsize()
                      atIndex:1];

    Constants params = {H, W, W + 1, 0};
    id<MTLBuffer> params_buf = [g_ctx.device newBufferWithBytes:&params
                                                         length:sizeof(Constants)
                                                        options:MTLResourceStorageModeShared];
    [computeEncoder setBuffer:params_buf offset:0 atIndex:2];

    MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((W + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                              (H + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height, 1);

    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  }
}

// ==========================================
// PHASE 1.5: Extract critical points
// ==========================================
gpu::CriticalPointsAsTensors launch_extract_critical_points_metal(torch::Tensor d_paired_with, int H, int W, int Nx) {
  @autoreleasepool {
    int num_cells = 4 * (H + 1) * (W + 1);

    size_t max_expected = num_cells / 3;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS);

    torch::Tensor d_vertices = torch::empty({(long)max_expected}, opts);
    torch::Tensor d_edges = torch::empty({(long)max_expected}, opts);
    torch::Tensor d_faces = torch::empty({(long)max_expected}, opts);
    torch::Tensor d_counters = torch::empty({3}, opts);

    at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    id<MTLCommandBuffer> commandBuffer = [g_ctx.commandQueue commandBuffer];

    // CRITICAL FIX: Initialize atomic counters to 0 natively on the GPU
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder fillBuffer:getMTLBufferStorage(d_counters)
                      range:NSMakeRange(d_counters.storage_offset() * d_counters.itemsize(),
                                        d_counters.numel() * d_counters.itemsize())
                      value:0x00];
    [blitEncoder endEncoding];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:g_ctx.extract_pipeline];
    [computeEncoder setBuffer:getMTLBufferStorage(d_paired_with)
                       offset:d_paired_with.storage_offset() * d_paired_with.itemsize()
                      atIndex:0];
    [computeEncoder setBuffer:getMTLBufferStorage(d_vertices)
                       offset:d_vertices.storage_offset() * d_vertices.itemsize()
                      atIndex:1];
    [computeEncoder setBuffer:getMTLBufferStorage(d_edges)
                       offset:d_edges.storage_offset() * d_edges.itemsize()
                      atIndex:2];
    [computeEncoder setBuffer:getMTLBufferStorage(d_faces)
                       offset:d_faces.storage_offset() * d_faces.itemsize()
                      atIndex:3];
    [computeEncoder setBuffer:getMTLBufferStorage(d_counters)
                       offset:d_counters.storage_offset() * d_counters.itemsize()
                      atIndex:4];

    Constants params = {H, W, Nx, 0};
    id<MTLBuffer> params_buf = [g_ctx.device newBufferWithBytes:&params
                                                         length:sizeof(Constants)
                                                        options:MTLResourceStorageModeShared];
    [computeEncoder setBuffer:params_buf offset:0 atIndex:5];

    int threads = 256;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threads, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((num_cells + threads - 1) / threads, 1, 1);

    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    torch::Tensor cpu_counters = d_counters.cpu();
    int v_count = cpu_counters[0].item<int>();
    int e_count = cpu_counters[1].item<int>();
    int f_count = cpu_counters[2].item<int>();

    torch::Tensor exact_vertices = d_vertices.slice(0, 0, v_count);
    torch::Tensor exact_edges = d_edges.slice(0, 0, e_count);
    torch::Tensor exact_faces = d_faces.slice(0, 0, f_count);

    return {exact_faces, exact_edges, exact_vertices};
  }
}

// ==========================================
// PHASE 2: Trace Saddles
// ==========================================
TracedSaddlesTensors launch_trace_from_saddles_metal(torch::Tensor d_data, torch::Tensor d_paired_with,
                                                     torch::Tensor d_saddles, int H, int W, int Nx, bool is_dual) {
  @autoreleasepool {
    auto sad_count = d_saddles.numel();
    if (sad_count == 0) return {};

    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS);
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS);

    TracedSaddlesTensors out;
    out.saddles = d_saddles;

    out.max_c1 = torch::full({sad_count}, -1, int_opts);
    out.max_c2 = torch::full({sad_count}, -1, int_opts);
    out.min_c1 = torch::full({sad_count}, -1, int_opts);
    out.min_c2 = torch::full({sad_count}, -1, int_opts);
    out.s_vals = torch::full({sad_count}, -1.0f, float_opts);
    out.max_len = torch::full({sad_count * 2}, 0, int_opts);
    out.min_len = torch::full({sad_count * 2}, 0, int_opts);

    at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    id<MTLCommandBuffer> commandBuffer = [g_ctx.commandQueue commandBuffer];

    // Bypass PyTorch's lazy allocator entirely. Create a raw Metal buffer
    // in Shared Memory, and synchronously memset it to 0 from the CPU.
    id<MTLBuffer> atomic_buf = [g_ctx.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    memset(atomic_buf.contents, 0, sizeof(uint32_t));

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    if (is_dual) {
      [computeEncoder setComputePipelineState:g_ctx.trace_pipeline_dual];
    } else {
      [computeEncoder setComputePipelineState:g_ctx.trace_pipeline_primal];
    }

    [computeEncoder setBuffer:getMTLBufferStorage(d_data) offset:d_data.storage_offset() * d_data.itemsize() atIndex:0];
    [computeEncoder setBuffer:getMTLBufferStorage(d_paired_with)
                       offset:d_paired_with.storage_offset() * d_paired_with.itemsize()
                      atIndex:1];
    [computeEncoder setBuffer:getMTLBufferStorage(out.saddles)
                       offset:out.saddles.storage_offset() * out.saddles.itemsize()
                      atIndex:2];
    [computeEncoder setBuffer:getMTLBufferStorage(out.max_c1)
                       offset:out.max_c1.storage_offset() * out.max_c1.itemsize()
                      atIndex:3];
    [computeEncoder setBuffer:getMTLBufferStorage(out.max_c2)
                       offset:out.max_c2.storage_offset() * out.max_c2.itemsize()
                      atIndex:4];
    [computeEncoder setBuffer:getMTLBufferStorage(out.min_c1)
                       offset:out.min_c1.storage_offset() * out.min_c1.itemsize()
                      atIndex:5];
    [computeEncoder setBuffer:getMTLBufferStorage(out.min_c2)
                       offset:out.min_c2.storage_offset() * out.min_c2.itemsize()
                      atIndex:6];
    [computeEncoder setBuffer:getMTLBufferStorage(out.s_vals)
                       offset:out.s_vals.storage_offset() * out.s_vals.itemsize()
                      atIndex:7];
    [computeEncoder setBuffer:getMTLBufferStorage(out.max_len)
                       offset:out.max_len.storage_offset() * out.max_len.itemsize()
                      atIndex:8];
    [computeEncoder setBuffer:getMTLBufferStorage(out.min_len)
                       offset:out.min_len.storage_offset() * out.min_len.itemsize()
                      atIndex:9];

    Constants params = {H, W, Nx, static_cast<int>(sad_count)};
    id<MTLBuffer> params_buf = [g_ctx.device newBufferWithBytes:&params
                                                         length:sizeof(Constants)
                                                        options:MTLResourceStorageModeShared];
    [computeEncoder setBuffer:params_buf offset:0 atIndex:10];

    // Bind our perfectly zeroed native Metal buffer to index 9
    [computeEncoder setBuffer:atomic_buf offset:0 atIndex:11];

    // LAUNCH THE MEGAKERNEL
    // 128 groups * 256 threads = 32,768 persistent threads circling the queue.
    // Here we don t have a thread per pixel but new saddles are given to threads that finish early.
    // So we just want to saturate the SMs.
    int threads = 256;
    int n_groups = 128;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threads, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake(n_groups, 1, 1);

    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    return out;
  }
}

// ==========================================
// PHASE 4 (V3): CELL GROUPS TRACING (Peaks/Basins)
// ==========================================
void launch_cell_groups_metal(torch::Tensor d_data, torch::Tensor d_paired_with, torch::Tensor d_fast_crit_map,
                              torch::Tensor d_uf_parent, torch::Tensor d_crits, torch::Tensor d_fast_region_id,
                              torch::Tensor d_out_groups, int H, int W, int Nx, bool is_dual, bool trace_faces) {
  @autoreleasepool {
    init_metal_context_if_needed();

    at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    id<MTLCommandBuffer> commandBuffer = [g_ctx.commandQueue commandBuffer];

    id<MTLBuffer> atomic_buf = [g_ctx.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    memset(atomic_buf.contents, 0, sizeof(uint32_t));

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    if (!trace_faces) {
      if (is_dual) {
        [computeEncoder setComputePipelineState:g_ctx.vertex_groups_pipeline_dual];
      } else {
        [computeEncoder setComputePipelineState:g_ctx.vertex_groups_pipeline_primal];
      }
    } else {
      if (is_dual) {
        [computeEncoder setComputePipelineState:g_ctx.face_groups_pipeline_dual];
      } else {
        [computeEncoder setComputePipelineState:g_ctx.face_groups_pipeline_primal];
      }
    }

    [computeEncoder setBuffer:getMTLBufferStorage(d_data) offset:d_data.storage_offset() * d_data.itemsize() atIndex:0];
    [computeEncoder setBuffer:getMTLBufferStorage(d_paired_with)
                       offset:d_paired_with.storage_offset() * d_paired_with.itemsize()
                      atIndex:1];
    [computeEncoder setBuffer:getMTLBufferStorage(d_fast_crit_map)
                       offset:d_fast_crit_map.storage_offset() * d_fast_crit_map.itemsize()
                      atIndex:2];
    [computeEncoder setBuffer:getMTLBufferStorage(d_uf_parent)
                       offset:d_uf_parent.storage_offset() * d_uf_parent.itemsize()
                      atIndex:3];
    [computeEncoder setBuffer:getMTLBufferStorage(d_crits)
                       offset:d_crits.storage_offset() * d_crits.itemsize()
                      atIndex:4];
    [computeEncoder setBuffer:getMTLBufferStorage(d_fast_region_id)
                       offset:d_fast_region_id.storage_offset() * d_fast_region_id.itemsize()
                      atIndex:5];
    [computeEncoder setBuffer:getMTLBufferStorage(d_out_groups)
                       offset:d_out_groups.storage_offset() * d_out_groups.itemsize()
                      atIndex:6];

    Constants params = {H, W, Nx, 0};
    id<MTLBuffer> params_buf = [g_ctx.device newBufferWithBytes:&params
                                                         length:sizeof(Constants)
                                                        options:MTLResourceStorageModeShared];
    [computeEncoder setBuffer:params_buf offset:0 atIndex:7];

    [computeEncoder setBuffer:atomic_buf offset:0 atIndex:8];

    // A 1D grid layout, 256*128 threads, cells are allocated dynamically via atomic counter
    int threads = 256;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threads, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake(128, 1, 1);

    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  }
}

// ==========================================
// PHASE 3: Trace Raw Arcs Geometry
// ==========================================
void launch_trace_raw_arcs_geometry_metal(torch::Tensor d_paired_with, torch::Tensor d_fast_crit_map,
                                          torch::Tensor d_crit_saddles, torch::Tensor d_max_offsets,
                                          torch::Tensor d_min_offsets, torch::Tensor d_flat_max,
                                          torch::Tensor d_flat_min, torch::Tensor d_saddle_nodes, int H, int W, int Nx,
                                          int num_saddles, bool trace_max_arcs, bool trace_min_arcs) {
  @autoreleasepool {
    init_metal_context_if_needed();

    at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    id<MTLCommandBuffer> commandBuffer = [g_ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:g_ctx.trace_raw_arcs_geometry_pipeline];

    // Bind all 8 PyTorch Tensors natively to the GPU buffers
    [computeEncoder setBuffer:getMTLBufferStorage(d_paired_with)
                       offset:d_paired_with.storage_offset() * d_paired_with.itemsize()
                      atIndex:0];
    [computeEncoder setBuffer:getMTLBufferStorage(d_fast_crit_map)
                       offset:d_fast_crit_map.storage_offset() * d_fast_crit_map.itemsize()
                      atIndex:1];
    [computeEncoder setBuffer:getMTLBufferStorage(d_crit_saddles)
                       offset:d_crit_saddles.storage_offset() * d_crit_saddles.itemsize()
                      atIndex:2];
    [computeEncoder setBuffer:getMTLBufferStorage(d_max_offsets)
                       offset:d_max_offsets.storage_offset() * d_max_offsets.itemsize()
                      atIndex:3];
    [computeEncoder setBuffer:getMTLBufferStorage(d_min_offsets)
                       offset:d_min_offsets.storage_offset() * d_min_offsets.itemsize()
                      atIndex:4];
    if (d_flat_max.defined()) {
      [computeEncoder setBuffer:getMTLBufferStorage(d_flat_max)
                         offset:d_flat_max.storage_offset() * d_flat_max.itemsize()
                        atIndex:5];
    } else {
      [computeEncoder setBuffer:nil offset:0 atIndex:5];
    }

    if (d_flat_min.defined()) {
      [computeEncoder setBuffer:getMTLBufferStorage(d_flat_min)
                         offset:d_flat_min.storage_offset() * d_flat_min.itemsize()
                        atIndex:6];
    } else {
      [computeEncoder setBuffer:nil offset:0 atIndex:6];
    }

    // The FFI Struct Buffer (UInt8 tensor mapped directly to the SaddleNode struct)
    [computeEncoder setBuffer:getMTLBufferStorage(d_saddle_nodes)
                       offset:d_saddle_nodes.storage_offset() * d_saddle_nodes.itemsize()
                      atIndex:7];

    // Pass constants
    Constants params = {H, W, Nx, num_saddles};
    id<MTLBuffer> params_buf = [g_ctx.device newBufferWithBytes:&params
                                                         length:sizeof(Constants)
                                                        options:MTLResourceStorageModeShared];
    [computeEncoder setBuffer:params_buf offset:0 atIndex:8];

    // Pass trace_max_arcs and trace_min_arcs
    bool trace_max_arcs_b = trace_max_arcs;
    id<MTLBuffer> trace_max_buf = [g_ctx.device newBufferWithBytes:&trace_max_arcs_b
                                                            length:sizeof(bool)
                                                           options:MTLResourceStorageModeShared];
    [computeEncoder setBuffer:trace_max_buf offset:0 atIndex:9];

    bool trace_min_arcs_b = trace_min_arcs;
    id<MTLBuffer> trace_min_buf = [g_ctx.device newBufferWithBytes:&trace_min_arcs_b
                                                            length:sizeof(bool)
                                                           options:MTLResourceStorageModeShared];
    [computeEncoder setBuffer:trace_min_buf offset:0 atIndex:10];

    // Simple 1D Grid Launch (1 thread per saddle)
    int threads = 256;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threads, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((num_saddles + threads - 1) / threads, 1, 1);

    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  }
}

// ==========================================
// PHASE 5: Simplification
// ==========================================
void launch_simplify_arcs_metal(torch::Tensor d_cancels, torch::Tensor d_init_t, torch::Tensor d_parent_ptrs,
                                torch::Tensor d_weights, torch::Tensor d_dag, torch::Tensor d_base_lens,
                                torch::Tensor d_alive, torch::Tensor d_pending, int num_cancels, int num_extrema,
                                int N2, int* host_dag_sz) {
  @autoreleasepool {
    init_metal_context_if_needed();

    // Ensure PyTorch is done touching any of these tensors before Metal takes over
    at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    id<MTLBuffer> d_ready_count = [g_ctx.device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> mtl_dag_sz = [g_ctx.device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

    ((int*)mtl_dag_sz.contents)[0] = *host_dag_sz;

    id<MTLBuffer> d_ready_list = [g_ctx.device newBufferWithLength:(num_cancels * sizeof(int))
                                                           options:MTLResourceStorageModePrivate];
    id<MTLBuffer> d_ready_R0 = [g_ctx.device newBufferWithLength:(num_cancels * sizeof(int))
                                                           options:MTLResourceStorageModePrivate];
    id<MTLBuffer> d_ready_R1 = [g_ctx.device newBufferWithLength:(num_cancels * sizeof(int))
                                                           options:MTLResourceStorageModePrivate];

    id<MTLBuffer> d_temp_weights = [g_ctx.device newBufferWithLength:(num_extrema * sizeof(uint64_t))
                                                             options:MTLResourceStorageModePrivate];
    id<MTLBuffer> d_temp_parents = [g_ctx.device newBufferWithLength:(num_extrema * sizeof(int))
                                                             options:MTLResourceStorageModePrivate];

    int threads = 256;
    int iteration = 0;

    while (iteration < num_cancels) {
      ((int*)d_ready_count.contents)[0] = 0;

      // ---------------------------------------------------------
      // KERNEL 1: EVALUATE
      // ---------------------------------------------------------
      id<MTLCommandBuffer> evalBuffer = [g_ctx.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> evalEncoder = [evalBuffer computeCommandEncoder];
      [evalEncoder setComputePipelineState:g_ctx.evaluate_simplification_pipeline];

      [evalEncoder setBuffer:getMTLBufferStorage(d_cancels) offset:0 atIndex:0];
      [evalEncoder setBuffer:getMTLBufferStorage(d_init_t) offset:0 atIndex:1];
      [evalEncoder setBuffer:getMTLBufferStorage(d_parent_ptrs) offset:0 atIndex:2];
      [evalEncoder setBuffer:getMTLBufferStorage(d_alive) offset:0 atIndex:3];
      [evalEncoder setBuffer:getMTLBufferStorage(d_pending) offset:0 atIndex:4];
      [evalEncoder setBuffer:d_ready_count offset:0 atIndex:5];
      [evalEncoder setBuffer:d_ready_list offset:0 atIndex:6];
      [evalEncoder setBuffer:d_ready_R0 offset:0 atIndex:7];
      [evalEncoder setBuffer:d_ready_R1 offset:0 atIndex:8];

      [evalEncoder setBytes:&num_cancels length:sizeof(int) atIndex:9];
      [evalEncoder setBytes:&num_extrema length:sizeof(int) atIndex:10];

      MTLSize tpt = MTLSizeMake(threads, 1, 1);
      MTLSize tpg_eval = MTLSizeMake((num_cancels + threads - 1) / threads, 1, 1);

      [evalEncoder dispatchThreadgroups:tpg_eval threadsPerThreadgroup:tpt];
      [evalEncoder endEncoding];

      [evalBuffer commit];
      [evalBuffer waitUntilCompleted];

      int ready = ((int*)d_ready_count.contents)[0];

      // printf("[Metal] Iteration %d: Found %d ready pairs\n", iteration, ready);

      if (ready == 0) {
        break;
      }

      // ---------------------------------------------------------
      // KERNEL 2: CONTRACT
      // ---------------------------------------------------------
      id<MTLCommandBuffer> contractBuffer = [g_ctx.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> contractEncoder = [contractBuffer computeCommandEncoder];
      [contractEncoder setComputePipelineState:g_ctx.contract_simplification_pipeline];

      [contractEncoder setBuffer:d_ready_list offset:0 atIndex:0];
      [contractEncoder setBuffer:d_ready_R0 offset:0 atIndex:1];
      [contractEncoder setBuffer:d_ready_R1 offset:0 atIndex:2];
      [contractEncoder setBuffer:getMTLBufferStorage(d_cancels) offset:0 atIndex:3];
      [contractEncoder setBuffer:getMTLBufferStorage(d_init_t) offset:0 atIndex:4];
      [contractEncoder setBuffer:getMTLBufferStorage(d_base_lens) offset:0 atIndex:5];
      [contractEncoder setBuffer:getMTLBufferStorage(d_parent_ptrs) offset:0 atIndex:6];
      [contractEncoder setBuffer:getMTLBufferStorage(d_weights) offset:0 atIndex:7];
      [contractEncoder setBuffer:getMTLBufferStorage(d_dag) offset:0 atIndex:8];
      [contractEncoder setBuffer:mtl_dag_sz offset:0 atIndex:9];
      [contractEncoder setBuffer:getMTLBufferStorage(d_alive) offset:0 atIndex:10];
      [contractEncoder setBuffer:getMTLBufferStorage(d_pending) offset:0 atIndex:11];

      [contractEncoder setBytes:&ready length:sizeof(int) atIndex:12];
      [contractEncoder setBytes:&N2 length:sizeof(int) atIndex:13];

      MTLSize tpg_contract = MTLSizeMake((ready + threads - 1) / threads, 1, 1);

      [contractEncoder dispatchThreadgroups:tpg_contract threadsPerThreadgroup:tpt];
      [contractEncoder endEncoding];

      [contractBuffer commit];
      [contractBuffer waitUntilCompleted];

      // ---------------------------------------------------------
      // KERNEL 3: COMPRESS PATHS
      // ---------------------------------------------------------
      bool use_alt = false;
      int pass = 0;
      MTLSize tpg_comp = MTLSizeMake((num_extrema + threads - 1) / threads, 1, 1);

      while (pass < 32) {
        ((int*)d_ready_count.contents)[0] = 0;  // Use ready_count as 'changed' flag

        id<MTLCommandBuffer> compBuffer = [g_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> compEncoder = [compBuffer computeCommandEncoder];
        [compEncoder setComputePipelineState:g_ctx.compress_paths_pipeline];

        id<MTLBuffer> p_in = use_alt ? d_temp_parents : getMTLBufferStorage(d_parent_ptrs);
        id<MTLBuffer> w_in = use_alt ? d_temp_weights : getMTLBufferStorage(d_weights);
        id<MTLBuffer> p_out = use_alt ? getMTLBufferStorage(d_parent_ptrs) : d_temp_parents;
        id<MTLBuffer> w_out = use_alt ? getMTLBufferStorage(d_weights) : d_temp_weights;

        [compEncoder setBuffer:p_in offset:0 atIndex:0];
        [compEncoder setBuffer:w_in offset:0 atIndex:1];
        [compEncoder setBuffer:p_out offset:0 atIndex:2];
        [compEncoder setBuffer:w_out offset:0 atIndex:3];
        [compEncoder setBuffer:getMTLBufferStorage(d_dag) offset:0 atIndex:4];
        [compEncoder setBuffer:mtl_dag_sz offset:0 atIndex:5];
        [compEncoder setBuffer:getMTLBufferStorage(d_base_lens) offset:0 atIndex:6];
        [compEncoder setBuffer:d_ready_count offset:0 atIndex:7];
        [compEncoder setBytes:&num_extrema length:sizeof(int) atIndex:8];
        [compEncoder setBytes:&N2 length:sizeof(int) atIndex:9];

        [compEncoder dispatchThreadgroups:tpg_comp threadsPerThreadgroup:tpt];
        [compEncoder endEncoding];

        [compBuffer commit];
        [compBuffer waitUntilCompleted];

        use_alt = !use_alt;
        pass++;

        int changed = ((int*)d_ready_count.contents)[0];
        if (changed == 0) break;
      }

      // If final data is in alt buffers, copy back to primary PyTorch tensors
      if (use_alt) {
        id<MTLCommandBuffer> copyBuffer = [g_ctx.commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [copyBuffer blitCommandEncoder];
        [blitEncoder copyFromBuffer:d_temp_parents
                       sourceOffset:0
                           toBuffer:getMTLBufferStorage(d_parent_ptrs)
                  destinationOffset:d_parent_ptrs.storage_offset() * d_parent_ptrs.itemsize()
                               size:num_extrema * sizeof(int)];
        [blitEncoder copyFromBuffer:d_temp_weights
                       sourceOffset:0
                           toBuffer:getMTLBufferStorage(d_weights)
                  destinationOffset:d_weights.storage_offset() * d_weights.itemsize()
                               size:num_extrema * sizeof(uint64_t)];
        [blitEncoder endEncoding];
        [copyBuffer commit];
        [copyBuffer waitUntilCompleted];
      }

      iteration++;
    }

    *host_dag_sz = ((int*)mtl_dag_sz.contents)[0];
  }
}