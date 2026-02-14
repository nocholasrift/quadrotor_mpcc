#ifndef QUADROTOR_MPCC_MPCC_CORE_H
#define QUADROTOR_MPCC_MPCC_CORE_H

#include <quadrotor_mpcc/quadrotor_mpcc.h>
#include <quadrotor_mpcc/types.h>

#include <map>
#include <variant>

namespace mpcc {

class MPCCore {

public:
  MPCCore();

  ~MPCCore();

  void load_params(const std::map<std::string, double> &params);

  std::array<double, 2> solve(const Eigen::VectorXd &state,
                              bool is_reverse = false);
  /***********************
   * Setters and Getters
   ***********************/
  // compiler was giving me linker errrors so I had to implement the fn here :(
  void set_odom(const Eigen::Vector3d &odom);
  void set_trajectory(const Eigen::VectorXd &x_pts,
                      const Eigen::VectorXd &y_pts,
                      const Eigen::VectorXd &knot_parameters);

  const bool get_solver_status() const;
  const Eigen::VectorXd &get_state() const;

  // AnyHorizon get_horizon() const;
  const std::map<std::string, double> &get_params() const;
  const types::Trajectory &get_trajectory() { return _trajectory; }
  const types::Trajectory &get_non_extended_trajectory() {
    return _non_extended_trajectory;
  }

private:
private:
  double _dt{0.1};
  double _max_anga{2 * M_PI};
  double _max_linacc{2.0};
  double _curr_vel{0.};
  double _curr_angvel{0.};
  double _max_vel{2.0};
  double _max_angvel{M_PI / 2.};

  int _mpc_steps{0};
  int _tube_degree{0};
  int _tube_samples{0};
  double _max_tube_width{0};
  double _tube_horizon{0};

  double _prop_gain{1.0};
  double _prop_angle_thresh{0.5};
  double _prev_s{0.};

  bool _is_tube_generated{false};
  bool _is_traj_set{false};
  bool _is_map_util_set{false};

  bool _use_cbf{false};
  bool _traj_reset{false};
  bool _has_run{false};

  types::Trajectory _trajectory;
  types::Trajectory _non_extended_trajectory;

  std::array<double, 2> _prev_cmd;
  Eigen::VectorXd _state;

  Eigen::Vector3d _odom{0., 0., 0.};
  Eigen::Vector2d _goal{0., 0.};

  std::map<std::string, double> _params;

  // std::unique_ptr<MPCBase> _mpc;
  QuadrotorMPCC _mpc;
};
} // namespace mpcc

#endif
