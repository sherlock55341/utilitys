/**
 * @file geo.hpp
 * @author Chunyuan Zhao (zhaochunyuan@stu.pku.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-05-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace geo {
/**
 * @brief 2D Point
 * 
 * @tparam T 
 */
template <typename T> class PointT {
  private:
    T x_, y_;

  public:
/**
 * @brief Construct a new Point T object
 * 
 * @param _x x coordinate
 * @param _y y coordinate
 */
    PointT(T _x = T(), T _y = T()) : x_(_x), y_(_y) {}
/**
 * @brief Return x
 * 
 * @return T 
 */
    T x() const { return x_; }
/**
 * @brief Return y
 * 
 * @return T 
 */
    T y() const { return y_; }
/**
 * @brief Set x
 * 
 * @param _x 
 */
    void setx(int _x) { x_ = _x; }
/**
 * @brief Set y
 * 
 * @param _y 
 */
    void sety(int _y) { y_ = _y; }
/**
 * @brief Overload operator ==
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator==(const PointT<T> &other) const {
        return x() == other.x() && y() == other.y();
    }
/**
 * @brief Overload operator !=
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator!=(const PointT<T> &other) const {
        return x() != other.x() || y() != other.y();
    }
/**
 * @brief Overload operator <
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator<(const PointT<T> &other) const {
        if (x() != other.x())
            return x() < other.x();
        return y() < other.y();
    }
/**
 * @brief Overload operator >
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator>(const PointT<T> &other) const {
        if (x() != other.x())
            return x() > other.x();
        return y() > other.y();
    }
};
/**
 * @brief Rectangle
 * 
 * @tparam T 
 */
template <typename T> class RectT {
  private:
    T lx_, ly_, hx_, hy_;

  public:
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Construct a new Rect T object
 * 
 * @param _lx low x
 * @param _ly low y
 * @param _hx high x
 * @param _hy high y
 */
    RectT(T _lx = T(), T _ly = T(), T _hx = T(), T _hy = T())
        : lx_(_lx), ly_(_ly), hx_(_hx), hy_(_hy) {
    }
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Construct a new Rect T object
 * 
 * @param p 
 */
    RectT(const PointT<T> &p)
        : lx_(p.x()), ly_(p.y()), hx_(p.x()), hy_(p.y()) {
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Return low x
 * 
 * @return T 
 */
        T
        lx() const {
        return lx_;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Return low y
 * 
 * @return T 
 */
        T
        ly() const {
        return ly_;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Return high x
 * 
 * @return T 
 */
        T
        hx() const {
        return hx_;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Return high y
 * 
 * @return T 
 */
        T
        hy() const {
        return hy_;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Set low x
 * 
 * @param _lx 
 */
        void
        setlx(T _lx) {
        lx_ = _lx;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Set low y
 * 
 * @param _ly 
 */
        void
        setly(T _ly) {
        ly_ = _ly;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Set high x
 * 
 * @param _hx 
 */
        void
        sethx(T _hx) {
        hx_ = _hx;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Set high y
 * 
 * @param _hy 
 */
        void
        sethy(T _hy) {
        hy_ = _hy;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Expand to x
 * 
 * @param x 
 */
        void
        updx(T x) {
        if (x < lx_)
            lx_ = x;
        if (x > hx_)
            hx_ = x;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
/**
 * @brief Expand to y
 * 
 * @param y 
 */
        void
        updy(T y) {
        if (y < ly_)
            ly_ = y;
        if (y > hy_)
            hy_ = y;
    }
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Expand with point
 * 
 * @param p 
 */
        void
        upd(const PointT<T> &p) {
        updx(p.x());
        updy(p.y());
    }
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Return half perimeter
 * 
 * @return T 
 */
        T
        hpwl() const {
        return (hx_ - lx_ + 1) + (hy_ - ly_ + 1);
    }
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Return area
 * 
 * @return T 
 */
        T
        area() const {
        return (hx_ - lx_ + 1) * (hy_ - ly_ + 1);
    }
};

/**
 * @brief Return if Rect lhs and Rect rhs has intersection
 * 
 * @tparam T 
 * @param lhs 
 * @param rhs 
 * @return true 
 * @return false 
 */
template <typename T>
bool hasIntersect(const RectT<T> &lhs, const RectT<T> &rhs) {
    return lhs.lx() <= rhs.hx() && lhs.hx() >= rhs.lx() &&
           lhs.ly() <= rhs.hy() && lhs.hy() >= rhs.ly();
}
/**
 * @brief Point on layer
 * 
 * @tparam T 
 */
template <typename T> class PointTOnLayer : public PointT<T> {
  private:
    T l_;

  public:
/**
 * @brief Construct a new Point T On Layer object
 * 
 * @param _l layer index
 * @param _x x coordinate
 * @param _y y coordinate
 */
    PointTOnLayer(T _l = T(), T _x = T(), T _y = T())
        : PointT<T>(_x, _y), l_(_l) {}
/**
 * @brief Return layer index
 * 
 * @return T 
 */
    T l() const { return l_; }
/**
 * @brief Set layer index
 * 
 * @param _l 
 */
    void setl(int _l) { l_ = _l; }
/**
 * @brief Overload operator ==
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator==(const PointTOnLayer<T> &other) const {
        return l() == other.l() && PointT<T>::x() == other.PointT<T>::x() &&
               PointT<T>::y() == other.PointT<T>::y();
    }
/**
 * @brief Overload operator !=
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator!=(const PointTOnLayer<T> &other) const {
        return l() != other.l() || PointT<T>::x() != other.PointT<T>::x() ||
               PointT<T>::y() != other.PointT<T>::y();
    }
/**
 * @brief Overload operator <
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator<(const PointTOnLayer<T> &other) const {
        if (l() != other.l())
            return l() < other.l();
        if (PointT<T>::x() != other.PointT<T>::x())
            return PointT<T>::x() < other.PointT<T>::x();
        return PointT<T>::y() < other.PointT<T>::y();
    }
/**
 * @brief Overload operator >
 * 
 * @param other 
 * @return true 
 * @return false 
 */
    bool operator>(const PointTOnLayer<T> &other) const {
        if (l() != other.l())
            return l() > other.l();
        if (PointT<T>::x() != other.PointT<T>::x())
            return PointT<T>::x() > other.PointT<T>::x();
        return PointT<T>::y() > other.PointT<T>::y();
    }
};
/**
 * @brief Rect on Layer
 * 
 * @tparam T 
 */
template <typename T> class RectTOnLayer : public RectT<T> {
  private:
    T l_;

  public:
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Construct a new Rect T On Layer object
 * 
 * @param _l layer index
 * @param _lx low x
 * @param _ly low y
 * @param _hx high x
 * @param _hy high y
 */
    RectTOnLayer(T _l = T(), T _lx = T(), T _ly = T(), T _hx = T(), T _hy = T())
        : RectT<T>(_lx, _ly, _hx, _hy), l_(_l) {
    }
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Construct a new Rect T On Layer< T> object
 * 
 * @param p point on layer 
 */
    RectTOnLayer<T>(const PointTOnLayer<T> &p)
        : RectT<T>(p.x(), p.y(), p.x(), p.y()), l_(p.l()) {
    }
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Return layer index
 * 
 * @return T 
 */
        T
        l() const {
        return l_;
    }
#ifdef __CUDACC__
    __device__ __host__
#endif
/**
 * @brief Set layer index
 * 
 * @param _l 
 */
        void
        setl(int _l) {
        l_ = _l;
    }
};
/**
 * @brief Interval
 * 
 * @tparam T 
 */
template <typename T> class IntvlT {
  private:
    T lo_, hi_;

  public:
/**
 * @brief Construct a new Intvl T object
 * 
 */
    IntvlT()
        : lo_(std::numeric_limits<T>::has_infinity
                  ? std::numeric_limits<T>::infinity()
                  : std::numeric_limits<T>::max()),
          hi_(std::numeric_limits<T>::has_infinity
                  ? -std::numeric_limits<T>::infinity()
                  : std::numeric_limits<T>::lowest()) {}
/**
 * @brief Construct a new Intvl T object
 * 
 * @param x [x, x]
 */
    IntvlT(T x) : lo_(x), hi_(x) {}
/**
 * @brief Construct a new Intvl T object
 * 
 * @param _lo low
 * @param _hi high
 */
    IntvlT(T _lo, T _hi) : lo_(_lo), hi_(_hi) {}
/**
 * @brief Return low
 * 
 * @return T 
 */
    T lo() const { return lo_; }
/**
 * @brief Return high
 * 
 * @return T 
 */
    T hi() const { return hi_; }
/**
 * @brief Return high - low + 1
 * 
 * @return T 
 */
    T len() const { return hi_ - lo_ + 1; }
/**
 * @brief Return high - low
 * 
 * @return T 
 */
    T range() const {return hi_ - lo_;}
/**
 * @brief Return (low + high) / 2
 * 
 * @return T 
 */
    T center() const { return (lo_ + hi_) / 2; }
/**
 * @brief Set low
 * 
 * @param _lo 
 */
    void setlo(T _lo) { lo_ = _lo; }
/**
 * @brief Set high
 * 
 * @param _hi 
 */
    void sethi(T _hi) { hi_ = _hi; }
/**
 * @brief Return if the interval is valid
 * 
 * @return true if valid
 * @return false if invalid
 */
    bool isValid() const { return lo_ <= hi_; }
/**
 * @brief Update the interval to include x
 * 
 * @param x 
 */
    void update(T x) {
        if (lo_ > x) {
            lo_ = x;
        }
        if (hi_ < x) {
            hi_ = x;
        }
    }
/**
 * @brief Return the intersection with other interval
 * 
 * @param other 
 * @return IntvlT<T> 
 */
    IntvlT<T> intersect(const IntvlT<T> &other) const {
        return IntvlT<T>(std::max(lo(), other.lo()),
                         std::min(hi(), other.hi()));
    }
};
} // namespace geo

UTILS_END