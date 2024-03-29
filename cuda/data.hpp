#pragma once

#include "api.hpp"

UTILS_BEGIN

namespace cuda {
template <typename T> class Data {
  private:
    T *host_, *device_;
    size_t size_, dsize_;

  public:
    Data() : host_(nullptr), device_(nullptr), size_(0), dsize_(0) {}

    Data(const std::vector<T> &other)
        : host_(nullptr), device_(nullptr), size_(0), dsize_(0) {
        mh(other.size());
        std::copy(other.begin(), other.end(), host_);
    }

    Data &mh(size_t N) {
        assert(N > 0);
        assert(host_ == nullptr);
        host_ = (T *)malloc(sizeof(T) * N);
        size_ = N;
        return *this;
    }

    Data &md(size_t N) {
        assert(N > 0);
        assert(device_ == nullptr);
        cuda::mallocDevice((void **)&device_, sizeof(T) * N);
        dsize_ = N;
        return *this;
    }

    Data &mhd(size_t N) {
        assert(N > 0);
        assert(host_ == nullptr);
        assert(device_ == nullptr);
        host_ = (T *)malloc(sizeof(T) * N);
        cuda::mallocDevice((void **)&device_, sizeof(T) * N);
        size_ = dsize_ = N;
        return *this;
    }

    void set(const T &val) const {
        assert(host_ != nullptr);
        std::fill(host_, host_ + size_, val);
    }

    void h2d() const {
        assert(host_ != nullptr);
        assert(device_ != nullptr);
        cuda::h2d(device_, host_, sizeof(T) * size_);
    }

    void d2h() const {
        assert(host_ != nullptr);
        assert(device_ != nullptr);
        cuda::d2h(host_, device_, sizeof(T) * dsize_);
    }

    void fh() {
        assert(host_ != nullptr);
        free(host_);
        host_ = nullptr;
        size_ = 0;
    }

    void fd() {
        assert(device_ != nullptr);
        cuda::freeDevice(device_);
        device_ = nullptr;
        dsize_ = 0;
    }

    void fhd() {
        fh();
        fd();
    }

    T *h() const { return host_; }

    T *d() const { return device_; }

    size_t size() const { return size_; }

    size_t dsize() const { return dsize_; }

    T &operator[](size_t x) {
        assert(x >= 0 && x < size_);
        return host_[x];
    }

    T operator[](size_t x) const {
        assert(x >= 0 && x < size_);
        return host_[x];
    }
};

namespace Transfer {
template <typename T> void cuda(T *&ptr, size_t num) {
    T *tmpPtr = nullptr;
    mallocDevice((void **)&tmpPtr, sizeof(T) * num);
    h2d(tmpPtr, ptr, sizeof(T) * num);
    free(ptr);
    ptr = tmpPtr;
}

template <typename T> void cpu(T *&ptr, size_t num) {
    T *tmpPtr = nullptr;
    tmpPtr = (T *)malloc(sizeof(T) * num);
    d2h(tmpPtr, ptr, sizeof(T) * num);
    freeDevice(ptr);
    ptr = tmpPtr;
}

template <typename T> void cuda(T *&dst, T *src, size_t num) {
    mallocDevice((void **)&dst, sizeof(T) * num);
    h2d(dst, src, sizeof(T) * num);
}

template <typename T> void cpu(T *&dst, T *src, size_t num) {
    dst = (T *)malloc(sizeof(T) * num);
    d2h(dst, src, sizeof(T) * num);
}
} // namespace Transfer

} // namespace cuda

UTILS_END