/**
 * @file algo.cpp
 * @author Chunyuan Zhao (zhaochunyuan@stu.pku.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-05-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "algo.hpp"

#include <mutex>
#include <thread>

UTILS_BEGIN

algo::UFS::UFS(int N) : par_(N) {
  assert(N > 0);
  for (int i = 0; i < par_.size(); i++) {
    par_[i] = i;
  }
}

void algo::UFS::reset(int N) {
  if (N == -1) {
    N = par_.size();
  }
  assert(N > 0);
  par_.resize(N);
  for (int i = 0; i < par_.size(); i++) {
    par_[i] = i;
  }
}

int algo::UFS::find(int x) {
  assert(x < par_.size() && x >= 0);
  return par_[x] == x ? x : par_[x] = find(par_[x]);
}

void algo::UFS::un(int x, int y) {
  assert(x < par_.size() && x >= 0);
  assert(y < par_.size() && y >= 0);
  par_[find(x)] = find(y);
}

UTILS_END