#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace algo {
class UFS {
 private:
  std::vector<int> par_;

 public:
  UFS(int N = 1);

  void reset(int N = -1);

  int find(int x);

  void un(int x, int y);
};

template <typename T>
void distinct(std::vector<T> &arr) {
  std::sort(arr.begin(), arr.end());
  arr.erase(std::unique(arr.begin(), arr.end()), arr.end());
}

template <typename T, class comp>
void distinct(std::vector<T> &arr, comp cmp) {
  std::sort(arr.begin(), arr.end(), cmp);
  arr.erase(std::unique(arr.begin(), arr.end()), arr.end());
}

enum MultiThreadStrategy { GREEDY, SEQ };

bool runMultiFunc(std::function<void(int)> func, int N, int numThreads,
                  MultiThreadStrategy strategy = SEQ);

bool runMultiFuncWithThreadIdx(std::function<void(int, int)> func, int N,
                               int numThreads,
                               MultiThreadStrategy strategy = SEQ);

}  // namespace algo

UTILS_END