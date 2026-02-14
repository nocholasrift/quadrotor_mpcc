#ifndef QUADROTOR_MPCC_TYPES_H
#define QUADROTOR_MPCC_TYPES_H

#include <iostream>
#include <stdexcept>

#include <Eigen/SVD>
#include <unsupported/Eigen/Splines>

namespace mpcc {
namespace types {

using StateHorizon = struct StateHorizon {
  Eigen::VectorXd xs;
  Eigen::VectorXd ys;
  Eigen::VectorXd arclens;
  Eigen::VectorXd arclens_dot;
};

using InputHorizon = struct InputHorizon {
  Eigen::VectorXd arclens_ddot;
};

template <typename Derived> struct MPCHorizon {
  typename Derived::StateHorizon states;
  typename Derived::InputHorizon inputs;
  unsigned int length{0};

  Eigen::VectorXd get_state_at_step(unsigned int step) const {
    return states.get_state_at_step(step);
  }

  Eigen::VectorXd get_input_at_step(unsigned int step) const {
    return inputs.get_input_at_step(step);
  }

  double get_arclen_at_step(unsigned int step) const {
    return states.arclens[step];
  }

  Eigen::VectorXd get_pos_at_step(unsigned int step) const {
    return static_cast<const typename Derived::MPCHorizon *>(this)->get_pos(
        step);
  }

  Eigen::VectorXd get_vel_at_step(unsigned int step) const {
    return static_cast<const typename Derived::MPCHorizon *>(this)->get_vel(
        step);
  }

  Eigen::VectorXd get_acc_at_step(unsigned int step) const {
    return static_cast<const typename Derived::MPCHorizon *>(this)->get_acc(
        step);
  }
};

template <typename MPCType> struct SolverTraits;

using Spline1D = Eigen::Spline<double, 1, 3>;

// polynomial class takes in coefficients in order of ascending degree!
// c0 + c1 * t + c2 * t^2 ...
class Polynomial {
public:
  using Coeffs = Eigen::VectorXd;

  Polynomial() = default;

  Polynomial(unsigned int degree)
      : coeffs_(Eigen::VectorXd::Zero(degree + 1)), degree_(degree) {}

  Polynomial(const Coeffs &coeffs)
      : coeffs_(coeffs), degree_(coeffs.size() - 1) {}

  Polynomial(const std::vector<double> &coeffs)
      : coeffs_(Eigen::Map<Coeffs>(const_cast<double *>(coeffs.data()),
                                   coeffs.size())),
        degree_(coeffs.size() - 1) {}

  ~Polynomial() = default;

  double pos(double t) const {
    Eigen::VectorXd basis = get_basis(t);
    return coeffs_.dot(basis);
  }

  double derivative(double t, unsigned int order) const {
    if (order == 0)
      return pos(t);
    if (order >= coeffs_.size())
      return 0.0;

    double result = 0.0;
    Eigen::VectorXd t_basis = get_basis(t);

    const unsigned int max_i = coeffs_.size() - order;
    for (unsigned int i = 0; i < max_i; ++i) {
      const unsigned int k = i + order;
      result += t_basis[i] * coeffs_[k] * deriv_coeff(k, order);
    }

    return result;
  }

  double operator()(double t) const { return pos(t); }
  double operator()(double t, unsigned int order) const {
    return derivative(t, order);
  }

  Eigen::VectorXd operator()(const Eigen::VectorXd &ts) const {
    Eigen::VectorXd vals(ts.size());
    for (int i = 0; i < ts.size(); ++i) {
      vals[i] = (*this)(ts[i]);
    }

    return vals;
  }

  const Coeffs &get_coeffs() const { return coeffs_; }

  // there are N+1 coefficients for an N degree polynomial
  const double get_degree() const { return coeffs_.size() - 1; }

  // i like being able to perform some operations in the setter so
  // i am also including an r-value setter.
  void set_coeffs(Coeffs &coeffs) {
    coeffs_ = coeffs;
    degree_ = coeffs.size() - 1;
  }

  void set_coeffs(Coeffs &&coeffs) {
    coeffs_ = coeffs;
    degree_ = coeffs.size() - 1;
  }

  // some static expressions for common orders, anything after 3,
  // just make your own local variable for it...
  static constexpr unsigned int kFirstOrder = 1;
  static constexpr unsigned int kSecondOrder = 2;
  static constexpr unsigned int kThirdOrder = 3;

private:
  Coeffs coeffs_;
  unsigned int degree_{0};

private:
  const Eigen::VectorXd get_basis(double t) const {

    double pow{1.};
    Eigen::VectorXd basis{Eigen::VectorXd::Zero(degree_ + 1)};

    for (unsigned int i{0}; i < basis.size(); ++i) {
      basis[i] = pow;
      pow *= t;
    }

    return basis;
  }

  const double deriv_coeff(unsigned int ind, unsigned int order) const {
    if (order > ind)
      return 0.;

    double c = 1.;
    for (unsigned int j = 0; j < order; ++j) {
      c *= static_cast<double>(ind - j);
    }

    return c;
  }
};

class Spline {
public:
  static constexpr unsigned int kDegree = 3;

  using Spline1D = Eigen::Spline<double, 1, kDegree>;
  using Point = Eigen::Vector2d;

  Spline() = default;

  Spline(const Eigen::RowVectorXd &knots, const Eigen::RowVectorXd &xs)
      : knots_(knots), xs_(xs) {

    const auto fitX = interp(xs, kDegree, knots);
    spline_ = Spline1D(fitX);
  }

  ~Spline() = default;

  double operator()(double t) const { return spline_(t).coeff(0); }

  double operator()(double t, unsigned int order) const {
    return spline_.derivatives(t, order).coeff(order);
  }

  double pos(double t) const { return spline_(t).coeff(0); }

  double derivative(double t, unsigned int order) const {
    if (order == 0) {
      return pos(t);
    }

    return spline_.derivatives(t, order).coeff(order);
  }

  const Eigen::RowVectorXd &get_knots() const { return knots_; }
  const Eigen::RowVectorXd &get_ctrls() const { return xs_; }

private:
  Spline1D interp(const Eigen::RowVectorXd &pts, Eigen::DenseIndex degree,
                  const Eigen::RowVectorXd &knot_parameters) {
    using namespace Eigen;

    typedef typename Spline1D::KnotVectorType::Scalar Scalar;
    typedef typename Spline1D::ControlPointVectorType ControlPointVectorType;

    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

    Eigen::RowVectorXd knots;
    knots.resize(knot_parameters.size() + degree + 1);

    // not-a-knot condition setup
    knots.segment(0, degree + 1) =
        knot_parameters(0) * Eigen::RowVectorXd::Ones(degree + 1);
    knots.segment(degree + 1, knot_parameters.size() - 4) =
        knot_parameters.segment(2, knot_parameters.size() - 4);
    knots.segment(knots.size() - degree - 1, degree + 1) =
        knot_parameters(knot_parameters.size() - 1) *
        Eigen::RowVectorXd::Ones(degree + 1);

    DenseIndex n = pts.cols();
    MatrixType A = MatrixType::Zero(n, n);
    for (DenseIndex i = 1; i < n - 1; ++i) {
      const DenseIndex span = Spline1D::Span(knot_parameters[i], degree, knots);

      // The segment call should somehow be told the spline order at compile
      // time.
      A.row(i).segment(span - degree, degree + 1) =
          Spline1D::BasisFunctions(knot_parameters[i], degree, knots);
    }
    A(0, 0) = 1.0;
    A(n - 1, n - 1) = 1.0;

    HouseholderQR<MatrixType> qr(A);

    // Here, we are creating a temporary due to an Eigen issue.
    ControlPointVectorType ctrls =
        qr.solve(MatrixType(pts.transpose())).transpose();

    return Spline1D(knots, ctrls);
  }

  Spline1D spline_;

  Eigen::RowVectorXd xs_;
  Eigen::RowVectorXd knots_;
};

class Trajectory {
public:
  using Point = Eigen::Vector2d;
  using Row = Eigen::RowVectorXd;

  enum class Side { kAbove, kBelow };
  enum class Indexing { Start, End };

  struct View {
    Row knots;
    Row xs;
    Row ys;
    double arclen{0};
  };

  Trajectory() = default;

  Trajectory(const Row &knots, const Row &xs, const Row &ys)
      : spline_x_(Spline(knots, xs)), spline_y_(Spline(knots, ys)),
        arclen_(knots(knots.size() - 1)) {}

  Trajectory(const Spline &x, const Spline &y)
      : spline_x_(x), spline_y_(y),
        arclen_(y.get_knots()(y.get_knots().size() - 1)) {}

  double get_arclen() const { return arclen_; }

  Point operator()(double s) const { return {spline_x_(s), spline_y_(s)}; }

  Point operator()(Indexing index) const {
    switch (index) {
    case Indexing::Start:
      return (*this)(0);
    case Indexing::End:
      return (*this)(arclen_);
    default:
      throw std::runtime_error("invalid index passed to trajectory.");
    }
  }

  // derivative
  Point operator()(double s, unsigned int order) const {
    return {spline_x_(s, order), spline_y_(s, order)};
  }

  Trajectory get_adjusted_traj(double s, unsigned int step_count = 0) const {
    if (s > arclen_) {
      throw std::runtime_error(
          "[Trajectory] start length passed to adjusted traj " +
          std::to_string(s) + " is larger than arc length" +
          std::to_string(arclen_));
    }

    if (step_count == 0) {
      step_count = spline_x_.get_knots().size();
    }

    Row knots, xs, ys;
    knots.resize(step_count);
    xs.resize(step_count);
    ys.resize(step_count);

    // get true end point of trajectory
    double epsilon = 0.05;
    Point end_point = (*this)(arclen_ - epsilon);

    // capture reference at each sample
    double ds = (arclen_ - s) / (step_count - 1);
    for (int i = 0; i < step_count; ++i) {
      knots(i) = ((double)i) * ds;

      Point p = (*this)(knots(i) + s);
      xs(i) = p(kX);
      ys(i) = p(kY);

      // std::cout << "p @ s=" << knots(i) + s << ": " << p.transpose() << "\n";
    }

    // std::cout << "[adjusted] xs[-1]=" << xs(step_count - 1) << "\n";
    // std::cout << "[adjusted] ys[-1]=" << ys(step_count - 1) << "\n";

    Spline x(knots, xs);
    Spline y(knots, ys);
    return Trajectory(x, y);
  }

  double get_closest_s(const Point &state) const {
    double s = 0;
    double min_dist = 1e6;
    Point pos{state(0), state(1)};
    for (double i = 0.0; i < arclen_; i += .01) {
      Point p = (*this)(i);
      double d = (pos - p).squaredNorm();
      if (d < min_dist) {
        min_dist = d;
        s = i;
      }
    }

    return s;
  }

  double distance_from(const Point &pt) const {
    double min_dist = 1e6;
    for (double i = 0.0; i < arclen_; i += .01) {
      double d = (pt - (*this)(i)).squaredNorm();
      if (d < min_dist) {
        min_dist = d;
      }
    }

    return min_dist;
  }

  Point get_unit_normal(double s, Side side = Side::kAbove) const {
    Point tangent = (*this)(s, kFirstOrder);
    tangent.normalize();

    switch (side) {
    case Side::kAbove:
      return Point(-tangent[kY], tangent[kX]);
    case Side::kBelow:
    default:
      return Point(tangent[kY], -tangent[kX]);
    }
  }

  View view() const {
    // x and y are forced to have same knots
    return {.knots = spline_x_.get_knots(),
            .xs = evaluate_axis_at_knots(kX),
            .ys = evaluate_axis_at_knots(kY),
            .arclen = arclen_};
  }

  const Row &get_ctrls_x() const { return spline_x_.get_ctrls(); }
  const Row &get_ctrls_y() const { return spline_y_.get_ctrls(); }

  static constexpr unsigned int kX = 0;
  static constexpr unsigned int kY = 1;

  static constexpr unsigned int kFirstOrder = 1;
  static constexpr unsigned int kSecondOrder = 2;

private:
  double arclen_{0};
  Spline spline_x_;
  Spline spline_y_;

private:
  // axis 0 == x, axis 1 == y
  // use kX and kY...
  Row evaluate_axis_at_knots(unsigned int axis) const {
    // spline x and y by construction have the same knots...
    Row knots = spline_x_.get_knots();
    Spline spl;
    if (axis != kX && axis != kY) {
      throw std::runtime_error(
          "[Trajectory] Invalid axis: " + std::to_string(axis) +
          " passed to get_axis_at_knots");
    }

    Row vals(knots.size());
    for (int i = 0; i < knots.size(); ++i) {
      double s = knots(i);
      vals[i] = axis == 0 ? spline_x_(s) : spline_y_(s);
    }

    return vals;
  }
};

class Corridor {
public:
  struct Sample {
    Eigen::Vector2d center, tangent, above, below;
  };

  enum class Side { kAbove, kBelow };

  Corridor(const Trajectory &ref, const Polynomial &abv, const Polynomial &blw,
           double s_start)
      : ref_(ref), abv_(abv), blw_(blw), s_offset_(s_start) {}

  const Polynomial &get_above_poly() const { return abv_; }

  const Polynomial &get_below_poly() const { return blw_; }

  const Trajectory &get_trajectory() const { return ref_; }

  const Polynomial::Coeffs &get_tube_coeffs(Side side) const {
    switch (side) {
    case Side::kAbove:
      return abv_.get_coeffs();
    case Side::kBelow:
      return blw_.get_coeffs();
    default:
      throw std::runtime_error(
          "Invalid parameter passed to get_tube_coeffs. Should be"
          "either Corridor::Side::kAbove or Corridor::Side::kBelow");
    }
  }

  Sample get_at(double s_local) const {
    double s_glob = s_offset_ + s_local;

    Eigen::Vector2d pos = ref_(s_glob);
    Eigen::Vector2d tan = ref_(s_glob, Trajectory::kFirstOrder);
    tan.normalize();

    Eigen::Vector2d norm(-tan.y(), tan.x());

    // tube borders are normalized in domain
    double d_abv = abv_.pos(s_local / ref_.get_arclen());
    double d_blw = blw_.pos(s_local / ref_.get_arclen());

    Sample corridor_sample;
    corridor_sample.center = pos;
    corridor_sample.tangent = tan;
    corridor_sample.above = pos + norm * d_abv;
    corridor_sample.below = pos + norm * d_blw;

    return corridor_sample;
  }

private:
  const Trajectory &ref_;
  const Polynomial &abv_;
  const Polynomial &blw_;
  double s_offset_;
};

} // namespace types
} // namespace mpcc

#endif
