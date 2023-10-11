#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace geo {
template <typename T> class PointT {
  private:
    T x_, y_;

  public:
    PointT(T _x = T(), T _y = T()) : x_(_x), y_(_y) {}

    T x() const { return x_; }

    T y() const { return y_; }

    void setx(int _x) { x_ = _x; }

    void sety(int _y) { y_ = _y; }

    bool operator==(const PointT<T> &other) const {
        return x() == other.x() && y() == other.y();
    }

    bool operator!=(const PointT<T> &other) const {
        return x() != other.x() || y() != other.y();
    }

    bool operator<(const PointT<T> &other) const {
        if (x() != other.x())
            return x() < other.x();
        return y() < other.y();
    }

    bool operator>(const PointT<T> &other) const {
        if (x() != other.x())
            return x() > other.x();
        return y() > other.y();
    }
};

template <typename T> class RectT {
  private:
    T lx_, ly_, hx_, hy_;

  public:
    RectT(T _lx = T(), T _ly = T(), T _hx = T(), T _hy = T())
        : lx_(_lx), ly_(_ly), hx_(_hx), hy_(_hy) {}

    RectT<T>(const PointT<T> &p)
        : lx_(p.x()), ly_(p.y()), hx_(p.x()), hy_(p.y()) {}

    T lx() const { return lx_; }

    T ly() const { return ly_; }

    T hx() const { return hx_; }

    T hy() const { return hy_; }

    void setlx(int _lx) { lx_ = _lx; }

    void setly(int _ly) { ly_ = _ly; }

    void sethx(int _hx) { hx_ = _hx; }

    void sethy(int _hy) { hy_ = _hy; }

    void updx(int x) {
        if (x < lx_)
            lx_ = x;
        if (x > hx_)
            hx_ = x;
    }

    void updy(int y) {
        if (y < ly_)
            ly_ = y;
        if (y > hy_)
            hy_ = y;
    }

    void upd(const PointT<T> &p) {
        updx(p.x());
        updy(p.y());
    }

    T hpwl() const { return (hx_ - lx_ + 1) + (hy_ - ly_ + 1);}

    T area() const { return (hx_ - lx_ + 1) * (hy_ - ly_ + 1); }
};

template<typename T>
bool hasIntersect(const RectT<T>& lhs, const RectT<T>& rhs){
    return lhs.lx() <= rhs.hx() && lhs.hx() >= rhs.lx() && lhs.ly() <= rhs.hy() && lhs.hy() >= rhs.ly();
}

template <typename T> class PointTOnLayer : public PointT<T> {
  private:
    T l_;

  public:
    PointTOnLayer(T _l = T(), T _x = T(), T _y = T())
        : PointT<T>(_x, _y), l_(_l) {}

    T l() const { return l_; }

    void setl(int _l) { l_ = _l; }

    bool operator==(const PointTOnLayer<T> &other) const {
        return l() == other.l() && PointT<T>::x() == other.PointT<T>::x() && PointT<T>::y() == other.PointT<T>::y();
    }

    bool operator!=(const PointTOnLayer<T> &other) const {
        return l() != other.l() || PointT<T>::x() != other.PointT<T>::x() || PointT<T>::y() != other.PointT<T>::y();
    }

    bool operator<(const PointTOnLayer<T> &other) const {
        if (l() != other.l())
            return l() < other.l();
        if (PointT<T>::x() != other.PointT<T>::x())
            return PointT<T>::x() < other.PointT<T>::x();
        return PointT<T>::y() < other.PointT<T>::y();
    }

    bool operator>(const PointTOnLayer<T> &other) const {
        if (l() != other.l())
            return l() > other.l();
        if (PointT<T>::x() != other.PointT<T>::x())
            return PointT<T>::x() > other.PointT<T>::x();
        return PointT<T>::y() > other.PointT<T>::y();
    }
};

template <typename T> class RectTOnLayer : public RectT<T> {
  private:
    T l_;

  public:
    RectTOnLayer(T _l = T(), T _lx = T(), T _ly = T(), T _hx = T(), T _hy = T())
        : RectT<T>(_lx, _ly, _hx, _hy), l_(_l) {}

    RectTOnLayer<T>(const PointTOnLayer<T> &p)
        : RectT<T>(p.x(), p.y(), p.x(), p.y()), l_(p.l()) {}

    T l() const { return l_; }

    void setl(int _l) { l_ = _l; }

    void updl(int l) { l_ = l; }
};

template <typename T> class IntvlT {
  private:
    T lo_, hi_;

  public:
    IntvlT(T _lo = T(), T _hi = T()) : lo_(_lo), hi_(_hi) {}

    IntvlT(T x) : lo_(x), hi_(x) {}

    T lo() const { return lo_; }

    T hi() const { return hi_; }

    T len() const { return hi_ - lo_ + 1; }

    T center() const { return (lo_ + hi_) / 2; }

    void setlo(T _lo) { lo_ = _lo; }

    void sethi(T _hi) { hi_ = _hi; }

    bool isValid() const { return lo_ <= hi_; }

    IntvlT<T> intersect(const IntvlT<T> &other) const {
        return IntvlT<T>(std::max(lo(), other.lo()),
                         std::min(hi(), other.hi()));
    }
};
} // namespace geo

UTILS_END