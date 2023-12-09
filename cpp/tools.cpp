#include "tools.hpp"
#include <iomanip>
#include <sstream>
#if defined(__unix__)
#include <sys/resource.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using bp = bg::model::point<int, 2, bg::cs::cartesian>;
using bb = bg::model::box<bp>;
using rt = bgi::rtree<std::pair<bb, int>, bgi::rstar<32>>;

UTILS_BEGIN

tools::Timer::Timer() : tp(std::chrono::high_resolution_clock::now()) {}

void tools::Timer::reset() { tp = std::chrono::high_resolution_clock::now(); }

double tools::Timer::elapse() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - tp)
               .count() /
           1e6;
}

tools::LogWithTimer::LogWithTimer(int _verbose) : verbose_(_verbose) {}

void tools::LogWithTimer::reset() { timer_.reset(); }

void tools::LogWithTimer::setVerbose(int _verbose) { verbose_ = _verbose; }

void tools::LogWithTimer::operator()(const char *__restrict__ format) const {
    if (verbose_ == 0) {
        return;
    }
    printf("[%7.3f s]", timer_.elapse());
    printf(format);
    fflush(stdout);
}

void tools::LogWithTimer::operator()(bool enable,
                                     const char *__restrict__ format) const {
    if (verbose_ == 0 || enable == 0) {
        return;
    }
    printf("[%7.3f s]", timer_.elapse());
    printf(format);
    fflush(stdout);
}

static void insert(rt &rtree, const geo::RectT<int> &box, int target) {
    bb bbox(bp(box.lx(), box.ly()), bp(box.hx(), box.hy()));
    rtree.insert({bbox, target});
}

static bool hasConflict(rt &rtree, const geo::RectT<int> &box, int target) {
    bb bbox(bp(box.lx(), box.ly()), bp(box.hx(), box.hy()));
    std::vector<std::pair<bb, int>> results;
    rtree.query(bgi::intersects(bbox), std::back_inserter(results));
    for (const auto &result : results) {
        if (result.second != target) {
            return true;
        }
    }
    return false;
}

std::vector<std::vector<int>>
tools::BatchGenerator::gen(const std::vector<geo::RectT<int>> &boxes,
                           const std::vector<int> &targets, int maxBatch,
                           int maxPace) {
    std::vector<std::vector<int>> batches;
    rt rtree;
    std::vector<bool> chosen(targets.size(), false);
    int last = 0;
    while (last < targets.size()) {
        rtree.clear();
        batches.emplace_back();
        auto &batch = batches.back();
        for (int i = last; i < targets.size(); i++) {
            if (batch.size() == maxBatch || i - last == maxPace)
                break;
            if (chosen[i])
                continue;
            if (!hasConflict(rtree, boxes[targets[i]], targets[i])) {
                insert(rtree, boxes[targets[i]], targets[i]);
                batch.emplace_back(targets[i]);
                chosen[i] = true;
            }
        }
        while (last < targets.size() && chosen[last]) {
            last++;
        }
    }
    return batches;
}

double tools::MemReporter::getCurrent(){
#if defined(__unix__)
    long rss = 0;
    FILE* fp = nullptr;
    if((fp = fopen("/proc/self/statm", "r")) == nullptr){
        return 0.0;
    }
    if(fscanf(fp, "%*s%ld", &rss) != 1){
        fclose(fp);
        return 0.0;
    }
    fclose(fp);
    return rss * sysconf(_SC_PAGESIZE) / 1048576.0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.WorkingSetSize / 1048576.0;
#else
    return 0.0;
#endif
}

double tools::MemReporter::getPeak(){
#if defined(__unix__)
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return rusage.ru_maxrss / 1024.0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.PeakWorkingSetSize / 1048576.0;
#else
    return 0.0;
#endif
}

UTILS_END