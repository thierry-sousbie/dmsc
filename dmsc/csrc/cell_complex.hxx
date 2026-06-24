#pragma once

#ifndef DMT_HOST_DEVICE
#ifdef __CUDACC__
#define DMT_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define DMT_HOST_DEVICE inline
#endif
#endif

// Discrete Morse cell complex navigation

/*
Cell_type Name    Geometry           Count       Index           Boundary (pixel coordinates)
3         Face    pixel corner       (W+1)x(H+1) (y * Nx + x)+3  (x,y) (x-1,y) (x,y-1) (x-1,y-1)
2         V-edge  edge btwn pixels   (W  )x(H+1) (y * Nx + x)+2  (x,y) (x,y-1)
1         H-edge  edge btwn pixels   (W+1)x(H  ) (y * Nx + x)+1  (x,y) (x-1,y)
0         Vertex  pixel              (W  )x(H  ) (y * Nx + x)+0  (x,y)
*/

DMT_HOST_DEVICE int cell_id(int type, int y, int x, int Nx) {
  return type + 4 * (y * Nx + x);
}

DMT_HOST_DEVICE int get_type(int id) {
  return id % 4;
}

DMT_HOST_DEVICE int get_y(int id, int Nx) {
  return (id / 4) / Nx;
}

DMT_HOST_DEVICE int get_x(int id, int Nx) {
  return (id / 4) % Nx;
}

DMT_HOST_DEVICE bool is_valid_v(int y, int x, int H, int W) {
  return y >= 0 && y < H && x >= 0 && x < W;
}

DMT_HOST_DEVICE bool is_valid_ehx(int y, int x, int H, int W) {
  return y >= 0 && y < H && x >= 0 && x <= W;
}

DMT_HOST_DEVICE bool is_valid_evy(int y, int x, int H, int W) {
  return y >= 0 && y <= H && x >= 0 && x < W;
}

DMT_HOST_DEVICE bool is_valid_f(int y, int x, int H, int W) {
  return y >= 0 && y <= H && x >= 0 && x <= W;
}
