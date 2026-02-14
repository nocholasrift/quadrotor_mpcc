#include <quadrotor_mpcc/mpcc_core.h>
#include <quadrotor_mpcc/termcolor.hpp>
#include <quadrotor_mpcc/utils.h>

#include <chrono>
#include <stdexcept>

using namespace mpcc;

MPCCore::MPCCore() {}

MPCCore::~MPCCore() {}

void MPCCore::load_params(const std::map<std::string, double> &params) {}

void MPCCore::set_odom(const Eigen::Vector3d &odom) { _odom = odom; }

void MPCCore::set_trajectory(const Eigen::VectorXd &x_pts,
                             const Eigen::VectorXd &y_pts,
                             const Eigen::VectorXd &knot_parameters) {}

std::array<double, 2> MPCCore::solve(const Eigen::VectorXd &state,
                                     bool is_reverse) {
  return {0., 0.};
}

const bool MPCCore::get_solver_status() const { return false; }

const Eigen::VectorXd &MPCCore::get_state() const { return _state; }

const std::map<std::string, double> &MPCCore::get_params() const {
  return _params;
}
