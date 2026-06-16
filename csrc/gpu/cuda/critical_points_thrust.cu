#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "../../cell_complex.hxx"

struct is_critical_vertex_functor {
  const int* paired;
  int H, W, Nx;
  is_critical_vertex_functor(const int* p, int h, int w, int nx) : paired(p), H(h), W(w), Nx(nx) {}

  __device__ bool operator()(const int& id) const {
    if (paired[id] != -1) return false;
    if (get_type(id) != 0) return false;
    return is_valid_v(get_y(id, Nx), get_x(id, Nx), H, W);
  }
};

struct is_critical_edge_functor {
  const int* paired;
  int H, W, Nx;
  is_critical_edge_functor(const int* p, int h, int w, int nx) : paired(p), H(h), W(w), Nx(nx) {}

  __device__ bool operator()(const int& id) const {
    if (paired[id] != -1) return false;
    int type = get_type(id);
    if (type == 1) return is_valid_ehx(get_y(id, Nx), get_x(id, Nx), H, W);
    if (type == 2) return is_valid_evy(get_y(id, Nx), get_x(id, Nx), H, W);
    return false;
  }
};

struct is_critical_face_functor {
  const int* paired;
  int H, W, Nx;
  is_critical_face_functor(const int* p, int h, int w, int nx) : paired(p), H(h), W(w), Nx(nx) {}

  __device__ bool operator()(const int& id) const {
    if (paired[id] != -1) return false;
    if (get_type(id) != 3) return false;
    return is_valid_f(get_y(id, Nx), get_x(id, Nx), H, W);
  }
};

void launch_extract_critical_points_cuda(const int* d_paired_with, int H, int W, int Nx,
                                         std::vector<int>& topological_mins, std::vector<int>& topological_saddles,
                                         std::vector<int>& topological_maxes, bool is_dual) {
  int num_cells = 4 * (H + 1) * (W + 1);
  int max_expected = num_cells / 5;

  auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

  // Physically named buffers
  torch::Tensor d_vertices = torch::empty({max_expected}, opts);
  torch::Tensor d_edges = torch::empty({max_expected}, opts);
  torch::Tensor d_faces = torch::empty({max_expected}, opts);

  thrust::device_ptr<int> t_vertices(d_vertices.data_ptr<int>());
  thrust::device_ptr<int> t_edges(d_edges.data_ptr<int>());
  thrust::device_ptr<int> t_faces(d_faces.data_ptr<int>());

  auto counting_iter = thrust::make_counting_iterator(0);

  auto vertex_end = thrust::copy_if(thrust::device, counting_iter, counting_iter + num_cells, t_vertices,
                                    is_critical_vertex_functor(d_paired_with, H, W, Nx));

  auto edge_end = thrust::copy_if(thrust::device, counting_iter, counting_iter + num_cells, t_edges,
                                  is_critical_edge_functor(d_paired_with, H, W, Nx));

  auto face_end = thrust::copy_if(thrust::device, counting_iter, counting_iter + num_cells, t_faces,
                                  is_critical_face_functor(d_paired_with, H, W, Nx));

  int vertex_count = vertex_end - t_vertices;
  int edge_count = edge_end - t_edges;
  int face_count = face_end - t_faces;

  // The O(1) topological routing switch
  if (is_dual) {
    // Dual: Faces are Minima, Vertices are Maxima
    topological_mins.resize(face_count);
    cudaMemcpy(topological_mins.data(), d_faces.data_ptr<int>(), face_count * sizeof(int), cudaMemcpyDeviceToHost);

    topological_saddles.resize(edge_count);
    cudaMemcpy(topological_saddles.data(), d_edges.data_ptr<int>(), edge_count * sizeof(int), cudaMemcpyDeviceToHost);

    topological_maxes.resize(vertex_count);
    cudaMemcpy(topological_maxes.data(), d_vertices.data_ptr<int>(), vertex_count * sizeof(int),
               cudaMemcpyDeviceToHost);
  } else {
    // Primal: Vertices are Minima, Faces are Maxima
    topological_mins.resize(vertex_count);
    cudaMemcpy(topological_mins.data(), d_vertices.data_ptr<int>(), vertex_count * sizeof(int), cudaMemcpyDeviceToHost);

    topological_saddles.resize(edge_count);
    cudaMemcpy(topological_saddles.data(), d_edges.data_ptr<int>(), edge_count * sizeof(int), cudaMemcpyDeviceToHost);

    topological_maxes.resize(face_count);
    cudaMemcpy(topological_maxes.data(), d_faces.data_ptr<int>(), face_count * sizeof(int), cudaMemcpyDeviceToHost);
  }
}
