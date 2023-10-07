#include "tools.hpp"

UTILS_BEGIN

tools::Timer::Timer() : tp(std::chrono::high_resolution_clock::now()) {}

void tools::Timer::reset(){
    tp = std::chrono::high_resolution_clock::now();
}

double tools::Timer::elapse() const{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tp).count() / 1e6;
}

tools::LogWithTimer::LogWithTimer(int _verbose) : verbose_(_verbose) {}

void tools::LogWithTimer::reset(){
    timer_.reset();
}

void tools::LogWithTimer::setVerbose(int _verbose){
    verbose_ = _verbose;
}

void tools::LogWithTimer::operator()(const char* __restrict__ format) const {
    if(verbose_ == 0){
        return ;
    }
    printf("[%7.3f s]", timer_.elapse());
    printf(format);
    fflush(stdout);
}

void tools::LogWithTimer::operator()(bool enable, const char* __restrict__ format) const {
    if(verbose_ == 0 || enable == 0){
        return ;
    }
    printf("[%7.3f s]", timer_.elapse());
    printf(format);
    fflush(stdout);
}


UTILS_END