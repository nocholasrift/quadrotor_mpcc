#ifndef QUADROTOR_MPCC_UTILS_H
#define QUADROTOR_MPCC_UTILS_H

#include <map>

#include <Eigen/Core>
#include <quadrotor_mpcc/types.h>

namespace utils {

inline std::ostream &operator<<(std::ostream &os,
                                const Eigen::VectorXd &vector) {
  os << "[";
  for (int i = 0; i < vector.size(); ++i) {
    os << vector[i];
    if (i < vector.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const Eigen::RowVectorXd &vector) {
  os << "[";
  for (int i = 0; i < vector.size(); ++i) {
    os << vector[i];
    if (i < vector.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1>
vector_to_eigen(const std::vector<T> &vec) {

  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>,
                    Eigen::Unaligned>(vec.data(), vec.size());
}

/**********************************************************************
 * Function: eval_traj
 * Description: Evaluates a polynomial at a given point
 * Parameters:
 * @param coeffs: const Eigen::VectorXd&
 * @param x: double
 * Returns:
 * double
 * Notes:
 * This function evaluates a polynomial (trajectory) at a given point
 **********************************************************************/
inline double eval_traj(const Eigen::VectorXd &coeffs, double x) {
  double ret = 0;
  double x_pow = 1;

  for (int i = 0; i < coeffs.size(); ++i) {
    ret += coeffs[i] * x_pow;
    x_pow *= x;
  }

  return ret;
}

inline void get_param(const std::map<std::string, double> &params,
                      const std::string &key, double &value) {
  if (auto it = params.find(key); it != params.end()) {
    value = params.at(key);
  }
}

inline mpcc::types::Trajectory
extend_trajectory(const mpcc::types::Trajectory &trajectory,
                  double extension_length) {
  using Trajectory = mpcc::types::Trajectory;

  Trajectory::View traj_view = trajectory.view();

  if (traj_view.arclen >= extension_length) {
    return trajectory;
  }

  const double epsilon = 0.1;
  size_t n_samples = traj_view.knots.size();
  double ds = extension_length / (n_samples - 1);
  double original_end_len = traj_view.arclen - epsilon;

  Trajectory::Point end_point = trajectory(original_end_len);
  Trajectory::Point end_deriv =
      trajectory(original_end_len, Trajectory::kFirstOrder);

  Trajectory::Row knots(n_samples), xs(n_samples), ys(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    double s = ds * i;
    knots(i) = s;

    // if s is within original bounds, use samples traj point
    // otherwise, pretend trajectory extends linearly from end point
    // in direction of end tangent vector.
    if (s < traj_view.arclen) {
      Trajectory::Point p = trajectory(s);
      xs(i) = p(Trajectory::kX);
      ys(i) = p(Trajectory::kY);
    } else {
      xs(i) = end_deriv(Trajectory::kX) * (s - original_end_len) +
              end_point(Trajectory::kX);
      ys(i) = end_deriv(Trajectory::kY) * (s - original_end_len) +
              end_point(Trajectory::kY);
    }
  }

  return Trajectory(knots, xs, ys);
}

} // namespace utils

#endif
