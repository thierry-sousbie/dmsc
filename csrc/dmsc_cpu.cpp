#include <ATen/record_function.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <vector>

#include "./cpu/dmsc_impl.hxx"
#include "./dmsc_struct.hxx"

using namespace cpu;

/*
Cell_type Name    Geometry           Count       Topology (PRIMAL)  Topology (DUAL)
3         Face    pixel corner       (W+1)x(H+1) Maximum            Minimum
2         V-edge  edge btwn pixels   (W  )x(H+1) Saddle             Saddle
1         H-edge  edge btwn pixels   (W+1)x(H  ) Saddle             Saddle
0         Vertex  pixel              (W  )x(H  ) Minimum            Maximum

note that the cell complex is defined as the dual of the image tesselation into a pixel grid. The
value of a cell is that of it highest adjacent vertex (pixel) by convention.

When IS_DUAL is true, cell types are reversed, and the value of a cell is given by that of its
lowest vertex and we sort in opposite direction. At the end, we swap minima and maxima, so we
compute the dual complex with the role of vertices/edges/faces reversed, i.e. they
represent maxima/saddles/minima respectively.
*/
template <bool IS_DUAL = false>
DMSComplex extract_single_dmsc_cpu_t(torch::Tensor scalar_field, float persistence_threshold, int block_size,
                                     bool return_gradient, bool trace_max_arcs, bool trace_min_arcs,
                                     bool trace_max_groups, bool trace_min_groups, cpu::Workspace& ws) {
  int H = ws.H;
  int W = ws.W;
  // int num_cells = ws.num_cells;
  ws.reset();

  tbb::global_control control(tbb::global_control::max_allowed_parallelism, at::get_num_threads());
  if (IS_DUAL) {
    std::swap(trace_max_arcs, trace_min_arcs);
    std::swap(trace_max_groups, trace_min_groups);
  }
  int num_blocks_y = (H + block_size - 1) / block_size;
  int num_blocks_x = (W + block_size - 1) / block_size;
  int total_blocks = num_blocks_y * num_blocks_x;

  compute_gradient_and_crit_points<IS_DUAL>(ws, scalar_field, total_blocks, num_blocks_x, block_size);
  trace_from_saddles<IS_DUAL>(ws, scalar_field);

  bool trace_any_arcs = trace_max_arcs || trace_min_arcs;
  if (trace_any_arcs) trace_raw_arcs_geometry(ws, trace_max_arcs, trace_min_arcs);

  compute_ppairs_and_simplify<IS_DUAL>(ws, persistence_threshold, trace_max_arcs, trace_min_arcs);

  if (trace_any_arcs) simplify_arcs_geometry(ws, trace_max_arcs, trace_min_arcs);
  if (trace_max_groups || trace_min_groups) compute_cell_groups(ws, trace_max_groups, trace_min_groups);

  return ws.complex<IS_DUAL>(return_gradient, trace_max_arcs, trace_min_arcs, trace_max_groups, trace_min_groups);
}

pybind11::object extract_dmsc_cpu(torch::Tensor scalar_field, float persistence_threshold, int block_size = 128,
                                  bool return_gradient = false, bool is_dual = false, bool trace_max_arcs = false,
                                  bool trace_min_arcs = false, bool trace_max_groups = false,
                                  bool trace_min_groups = false) {
  TORCH_CHECK(scalar_field.dim() == 2 || scalar_field.dim() == 3, "Input tensor must be 2D [H, W] or 3D [B, H, W]");

  scalar_field = scalar_field.contiguous();

  if (scalar_field.dim() == 2) {
    int H = scalar_field.size(0);
    int W = scalar_field.size(1);

    DMSComplex result;
    cpu::Workspace ws(H, W);
    if (is_dual) {
      result = extract_single_dmsc_cpu_t<true>(scalar_field, persistence_threshold, block_size, return_gradient,
                                               trace_max_arcs, trace_min_arcs, trace_max_groups, trace_min_groups, ws);
    } else {
      result = extract_single_dmsc_cpu_t<false>(scalar_field, persistence_threshold, block_size, return_gradient,
                                                trace_max_arcs, trace_min_arcs, trace_max_groups, trace_min_groups, ws);
    }

    return pybind11::cast(result);

  } else {
    // 3D Batched case
    int B = scalar_field.size(0);
    int H = scalar_field.size(1);
    int W = scalar_field.size(2);

    std::vector<DMSComplex> results;
    results.reserve(B);
    cpu::Workspace ws(H, W);

    for (int b = 0; b < B; ++b) {
      if (is_dual) {
        results.push_back(extract_single_dmsc_cpu_t<true>(scalar_field[b], persistence_threshold, block_size,
                                                          return_gradient, trace_max_arcs, trace_min_arcs,
                                                          trace_max_groups, trace_min_groups, ws));
      } else {
        results.push_back(extract_single_dmsc_cpu_t<false>(scalar_field[b], persistence_threshold, block_size,
                                                           return_gradient, trace_max_arcs, trace_min_arcs,
                                                           trace_max_groups, trace_min_groups, ws));
      }
    }

    return pybind11::cast(results);
  }
}

PYBIND11_MODULE(dmsc_cpu, m) {
  pybind11::class_<DMSComplex>(m, "DMSComplex", pybind11::module_local(), "...")
      .def_readonly("shape", &DMSComplex::shape)
      .def_readonly("max_pts", &DMSComplex::max_pts)
      .def_readonly("min_pts", &DMSComplex::min_pts)
      .def_readonly("sad_pts", &DMSComplex::sad_pts)
      .def_readonly("e_max", &DMSComplex::e_max)
      .def_readonly("e_min", &DMSComplex::e_min)
      .def_readonly("p_max", &DMSComplex::p_max)
      .def_readonly("p_min", &DMSComplex::p_min)
      .def_readonly("ppairs_max", &DMSComplex::ppairs_max)
      .def_readonly("ppairs_min", &DMSComplex::ppairs_min)
      .def_readonly("grad_indices", &DMSComplex::grad_indices)
      .def_readonly("peaks", &DMSComplex::peaks)
      .def_readonly("ridges", &DMSComplex::ridges)
      .def_readonly("ridge_offsets", &DMSComplex::ridge_offsets)
      .def_readonly("ridge_arc_offsets", &DMSComplex::ridge_arc_offsets)
      .def_readonly("basins", &DMSComplex::basins)
      .def_readonly("valleys", &DMSComplex::valleys)
      .def_readonly("valley_offsets", &DMSComplex::valley_offsets)
      .def_readonly("valley_arc_offsets", &DMSComplex::valley_arc_offsets)
      .def("__getitem__",
           [](const DMSComplex& r, int idx) {
             switch (idx) {
               case 0:
                 return r.shape;
               case 1:
                 return r.max_pts;
               case 2:
                 return r.min_pts;
               case 3:
                 return r.sad_pts;
               case 4:
                 return r.e_max;
               case 5:
                 return r.e_min;
               case 6:
                 return r.p_max;
               case 7:
                 return r.p_min;
               case 8:
                 return r.ppairs_max;
               case 9:
                 return r.ppairs_min;
               case 10:
                 return r.grad_indices;
               case 11:
                 return r.peaks;
               case 12:
                 return r.ridges;
               case 13:
                 return r.ridge_offsets;
               case 14:
                 return r.ridge_arc_offsets;
               case 15:
                 return r.basins;
               case 16:
                 return r.valleys;
               case 17:
                 return r.valley_offsets;
               case 18:
                 return r.valley_arc_offsets;
               default:
                 throw pybind11::index_error();
             }
           })
      .def("__len__", [](const DMSComplex& r) { return 19; });

  // Only one function exported now!
  m.def("extract_dmsc", &extract_dmsc_cpu, "Multithreaded Discrete Morse-Smale complex computation",
        pybind11::arg("scalar_field"), pybind11::arg("persistence_threshold"), pybind11::arg("block_size") = 128,
        pybind11::arg("return_gradient") = false, pybind11::arg("is_dual") = false,
        pybind11::arg("trace_max_arcs") = false, pybind11::arg("trace_min_arcs") = false,
        pybind11::arg("trace_max_groups") = false, pybind11::arg("trace_min_groups") = false);
}