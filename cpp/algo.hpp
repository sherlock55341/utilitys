/**
 * @file algo.hpp
 * @author Chunyuan Zhao (zhaochunyuan@stu.pku.edu.cn)
 * @brief Commonly Used Algorithms on CPU
 * @version 0.1
 * @date 2024-05-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace algo {    
/**
 * @brief Union Find Set
 * 
 */
class UFS {
 private:
  std::vector<int> par_;

 public:
 /**
  * @brief Construct a new UFS object
  * 
  * @param N Number of elements
  */
  UFS(int N = 1);
/**
 * @brief Reset the UFS object
 * 
 * @param N Number of elements
 */
  void reset(int N = -1);
/**
 * @brief Find the set that x belongs to
 * 
 * @param x 
 * @return int 
 */
  int find(int x);
/**
 * @brief merge x and y
 * 
 * @param x 
 * @param y 
 */
  void un(int x, int y);
};
/**
 * @brief Remove duplicates from array arr
 * 
 * @tparam T 
 * @param arr 
 */
template <typename T>
void distinct(std::vector<T> &arr) {
  std::sort(arr.begin(), arr.end());
  arr.erase(std::unique(arr.begin(), arr.end()), arr.end());
}
/**
 * @brief Remove duplicates from array arr with cmp
 * 
 * @tparam T 
 * @tparam comp 
 * @param arr 
 * @param cmp 
 */
template <typename T, class comp>
void distinct(std::vector<T> &arr, comp cmp) {
  std::sort(arr.begin(), arr.end(), cmp);
  arr.erase(std::unique(arr.begin(), arr.end()), arr.end());
}
}  // namespace algo

UTILS_END