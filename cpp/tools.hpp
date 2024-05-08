/**
 * @file tools.hpp
 * @author Chunyuan Zhao (zhaochunyuan@stu.pku.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-05-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include "geo.hpp"
#include <chrono>

#include "../global.hpp"

UTILS_BEGIN

namespace tools {
/**
 * @brief Timer
 * 
 */
class Timer {
  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> tp;

  public:
/**
 * @brief Construct a new Timer object
 * 
 */
    Timer();
/**
 * @brief Reset the timer
 * 
 */
    void reset();
/**
 * @brief Return time from reset / construction to now
 * 
 * @return double unit : second
 */
    double elapse() const;
};
/**
 * @brief A log printer with timer
 * 
 */
class LogWithTimer {
  private:
    Timer timer_;
    int verbose_;

  public:
/**
 * @brief Construct a new Log With Timer object
 * 
 * @param _verbose if print log
 */
    LogWithTimer(int _verbose = 1);
/**
 * @brief Reset the timer
 * 
 */
    void reset();
/**
 * @brief Set the Verbose object
 * 
 * @param _verbose 
 */
    void setVerbose(int _verbose);
/**
 * @brief Print log
 * 
 * @tparam Args 
 * @param format 
 * @param args 
 */
    template <typename... Args>
    void operator()(const char *__restrict__ format, Args &&...args) const {
        if (verbose_ == 0) {
            return;
        }
        printf("[%7.3f s]", timer_.elapse());
        printf(format, args...);
        fflush(stdout);
    }
/**
 * @brief Print log
 * 
 * @param format 
 */
    void operator()(const char *__restrict__ format) const;
/**
 * @brief Print log
 * 
 * @tparam Args 
 * @param enable 
 * @param format 
 * @param args 
 */
    template <typename... Args>
    void operator()(bool enable, const char *__restrict__ format,
                    Args &&...args) const {
        if (verbose_ == 0 || enable == 0) {
            return;
        }
        printf("[%7.3f s]", timer_.elapse());
        printf(format, args...);
        fflush(stdout);
    }
/**
 * @brief Print log
 * 
 * @param enable 
 * @param format 
 */
    void operator()(bool enable, const char *__restrict__ format) const;
};

/**
 * @brief Memory usage reporter
 * 
 */
class MemReporter{
public:
/**
 * @brief Return current memory usage (unit : MB)
 * 
 * @return double 
 */
    static double getCurrent();
/**
 * @brief Return peak memory usage (unit : MB)
 * 
 * @return double 
 */
    static double getPeak();
};
} // namespace tools

UTILS_END