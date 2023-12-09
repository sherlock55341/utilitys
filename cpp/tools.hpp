#pragma once

#include "geo.hpp"
#include <chrono>

#include "../global.hpp"

UTILS_BEGIN

namespace tools {
class Timer {
  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> tp;

  public:
    Timer();

    void reset();

    double elapse() const;
};

class LogWithTimer {
  private:
    Timer timer_;
    int verbose_;

  public:
    LogWithTimer(int _verbose = 1);

    void reset();

    void setVerbose(int _verbose);

    template <typename... Args>
    void operator()(const char *__restrict__ format, Args &&...args) const {
        if (verbose_ == 0) {
            return;
        }
        printf("[%7.3f s]", timer_.elapse());
        printf(format, args...);
        fflush(stdout);
    }

    void operator()(const char *__restrict__ format) const;

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

    void operator()(bool enable, const char *__restrict__ format) const;
};

//Unit : MB
class MemReporter{
public:
    static double getCurrent();
    static double getPeak();
};

class BatchGenerator {
  public:

    static std::vector<std::vector<int>> gen(const std::vector<geo::RectT<int>> &boxes,
                                             const std::vector<int> &targets,
                                             int maxBatch, int maxPace);
};
} // namespace tools

UTILS_END