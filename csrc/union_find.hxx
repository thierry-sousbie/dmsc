#pragma once

#include <vector>

struct UnionFind {
  std::vector<int> parent;

  UnionFind(int n) {
    parent.resize(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
  }

  int find(int i) {
    if (i == -1) return -1;

    int root = i;
    while (root != -1 && parent[root] != root) {
      root = parent[root];
    }

    // path compression
    int curr = i;
    while (curr != root && curr != -1) {
      int nxt = parent[curr];
      parent[curr] = root;
      curr = nxt;
    }

    return root;
  }

  void unite(int i, int j) {
    int root_i = find(i);
    int root_j = find(j);
    unite_from_root(root_i, root_j);
  }

  void unite_from_root(int root_i, int root_j) {
    // If root_i is the Void (-1), it can never be swallowed. Do nothing.
    // Otherwise, point root_i to root_j (which is safely allowed to be -1).
    if (root_i != root_j && root_i != -1) {
      parent[root_i] = root_j;
    }
  }
};