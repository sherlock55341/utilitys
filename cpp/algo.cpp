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

bool algo::runMultiFunc(std::function<void(int)> func, int N, int numThreads,
                        MultiThreadStrategy strategy) {
  numThreads = std::min(N, numThreads);
  if (numThreads <= 0) {
    assert(numThreads > 0);
    return false;
  }
  if (numThreads == 1) {
    for (int i = 0; i < N; i++) {
      func(i);
    }
    return true;
  }
  if (strategy == GREEDY) {
    std::mutex mtx;
    std::vector<std::thread> ths(numThreads);
    int globalJobIdx = 0;
    auto threadFunc = [&](int threadIdx) {
      int localJobIdx;
      while (true) {
        mtx.lock();
        localJobIdx = globalJobIdx++;
        mtx.unlock();
        if (localJobIdx >= N) {
          break;
        }
        func(localJobIdx);
      }
    };

    for (int i = 0; i < numThreads; i++) {
      ths[i] = std::thread(threadFunc, i);
    }
    for (int i = 0; i < numThreads; i++) {
      ths[i].join();
    }
    return true;
  } else {
    std::vector<std::thread> ths(numThreads);
    auto threadFunc = [&](int threadIdx) {
      for (int i = threadIdx; i < N; i += numThreads) {
        func(i);
      }
    };

    for (int i = 0; i < numThreads; i++) {
      ths[i] = std::thread(threadFunc, i);
    }
    for (int i = 0; i < numThreads; i++) {
      ths[i].join();
    }
    return true;
  }
  return false;
}

bool algo::runMultiFuncWithThreadIdx(std::function<void(int, int)> func, int N,
                                     int numThreads,
                                     MultiThreadStrategy strategy) {
  numThreads = std::min(N, numThreads);
  if (numThreads <= 0) {
    assert(numThreads > 0);
    return false;
  }
  if (numThreads == 1) {
    for (int i = 0; i < N; i++) {
      func(i, 0);
    }
    return true;
  }
  if (strategy == GREEDY) {
    std::mutex mtx;
    std::vector<std::thread> ths(numThreads);
    int globalJobIdx = 0;
    auto threadFunc = [&](int threadIdx) {
      int localJobIdx;
      while (true) {
        mtx.lock();
        localJobIdx = globalJobIdx++;
        mtx.unlock();
        if (localJobIdx >= N) {
          break;
        }
        func(localJobIdx, threadIdx);
      }
    };

    for (int i = 0; i < numThreads; i++) {
      ths[i] = std::thread(threadFunc, i);
    }
    for (int i = 0; i < numThreads; i++) {
      ths[i].join();
    }
    return true;
  } else {
    std::vector<std::thread> ths(numThreads);
    auto threadFunc = [&](int threadIdx) {
      for (int i = threadIdx; i < N; i += numThreads) {
        func(i, threadIdx);
      }
    };

    for (int i = 0; i < numThreads; i++) {
      ths[i] = std::thread(threadFunc, i);
    }
    for (int i = 0; i < numThreads; i++) {
      ths[i].join();
    }
    return true;
  }
  return false;
}

UTILS_END