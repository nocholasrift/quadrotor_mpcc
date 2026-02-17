#include "mpcc/ros/mpcc_ros.h"
#include "mpcc/common/mpcc_core.h"

#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_msgs/GridMap.h>
#include <math.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <tf/tf.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <algorithm>
#include <unordered_map>

#include "nav_msgs/OccupancyGrid.h"
#include "ros/console.h"
#include <grid_map_ros/GridMapRosConverter.hpp>

MPCCROS::MPCCROS(ros::NodeHandle &nh) : _nh("~") {
  _is_init = false;
  _is_goal = false;

  _vel_msg.linear.x = 0;
  _vel_msg.angular.z = 0;

  double freq;

  // MPC params
  _nh.param("vel_pub_freq", _vel_pub_freq, 20.0);
  _nh.param("controller_frequency", freq, 10.0);
  _nh.param("mpc_steps", _mpc_steps, 10.0);

  // param cant do unsigned ints?
  int input_type;
  _nh.param("mpc_input_type", input_type,
            static_cast<int>(mpcc::MPCType::kDoubleIntegrator));
  _mpc_input_type = static_cast<mpcc::MPCType>(input_type);

  // Cost function params
  _nh.param("w_vel", _w_vel, 1.0);
  _nh.param("w_angvel", _w_angvel, 1.0);
  _nh.param("w_linvel", _w_linvel, 1.0);
  _nh.param("w_angvel_d", _w_angvel_d, 1.0);
  _nh.param("w_linvel_d", _w_linvel_d, .5);
  _nh.param("w_etheta", _w_etheta, 1.0);
  _nh.param("w_cte", _w_cte, 1.0);

  _nh.param("w_lag_e", _w_ql, 50.0);
  _nh.param("w_contour_e", _w_qc, .1);
  _nh.param("w_speed", _w_q_speed, .3);

  // Constraint params
  _nh.param("max_angvel", _max_angvel, 3.0);
  _nh.param("max_linvel", _max_linvel, 2.0);
  _nh.param("max_linacc", _max_linacc, 3.0);
  _nh.param("max_angacc", _max_anga, 2 * M_PI);
  _nh.param("min_alpha", _min_alpha, .1);
  _nh.param("max_alpha", _max_alpha, 10.);
  _nh.param("min_alpha_dot", _min_alpha_dot, -1.0);
  _nh.param("max_alpha_dot", _max_alpha_dot, 1.0);
  _nh.param("min_h_val", _min_h_val, -1e8);
  _nh.param("max_h_val", _max_h_val, 1e8);

  _nh.param("bound_value", _bound_value, 1.0e19);

  // Goal params
  _nh.param("x_goal", _x_goal, 0.0);
  _nh.param("y_goal", _y_goal, 0.0);
  _nh.param("goal_tolerance", _tol, 0.3);

  // Teleop params
  _nh.param("teleop", _teleop, false);
  _nh.param<std::string>("frame_id", _frame_id, "odom");

  // clf params
  _nh.param("w_lyap_lag_e", _w_ql_lyap, 1.0);
  _nh.param("w_lyap_contour_e", _w_qc_lyap, 1.0);
  _nh.param("clf_gamma", _clf_gamma, .5);

  // cbf params
  _nh.param("use_cbf", _use_cbf, false);
  _nh.param("cbf_alpha_abv", _cbf_alpha_abv, .5);
  _nh.param("cbf_alpha_blw", _cbf_alpha_blw, .5);
  _nh.param("cbf_colinear", _cbf_colinear, .1);
  _nh.param("cbf_padding", _cbf_padding, .1);
  _nh.param("dynamic_alpha", _use_dynamic_alpha, false);

  // proportional controller params
  _nh.param("prop_gain", _prop_gain, .5);
  _nh.param("prop_gain_thresh", _prop_angle_thresh, 30. * M_PI / 180.);

  // tube parameters
  _nh.param("tube_poly_degree", _tube_degree, 6);
  _nh.param("tube_num_samples", _tube_samples, 50);
  _nh.param("max_tube_width", _max_tube_width, 2.0);

  _nh.param("ref_length_size", _mpc_ref_len_sz, 4.);
  _nh.param("mpc_ref_samples", _mpc_ref_samples, 10);

  _nh.param("task_id", _task_id, -1);
  _nh.param("is_eval", _is_eval, false);
  _nh.param("logging", _is_logging, false);
  _nh.param("num_samples", _num_samples, static_cast<int>(1e6));
  _nh.param("max_path_length", _max_path_length, static_cast<int>(1e6));

  _dt = 1.0 / freq;

  _mpc_params["DT"] = _dt;
  _mpc_params["STEPS"] = _mpc_steps;
  _mpc_params["W_V"] = _w_linvel;
  _mpc_params["W_ANGVEL"] = _w_angvel;
  _mpc_params["W_DA"] = _w_linvel_d;
  _mpc_params["W_DANGVEL"] = _w_angvel_d;
  _mpc_params["W_ETHETA"] = _w_etheta;
  _mpc_params["W_POS"] = _w_pos;
  _mpc_params["W_CTE"] = _w_cte;
  _mpc_params["LINVEL"] = _max_linvel;
  _mpc_params["ANGVEL"] = _max_angvel;
  _mpc_params["BOUND"] = _bound_value;
  _mpc_params["X_GOAL"] = _x_goal;
  _mpc_params["Y_GOAL"] = _y_goal;

  _mpc_params["ANGLE_THRESH"] = _prop_angle_thresh;
  _mpc_params["ANGLE_GAIN"] = _prop_gain;

  _mpc_params["W_LAG"] = _w_ql;
  _mpc_params["W_CONTOUR"] = _w_qc;
  _mpc_params["W_SPEED"] = _w_q_speed;

  _mpc_params["REF_LENGTH"] = _mpc_ref_len_sz;
  _mpc_params["REF_SAMPLES"] = _mpc_ref_samples;

  _mpc_params["CLF_GAMMA"] = _clf_gamma;
  _mpc_params["CLF_W_LAG"] = _w_ql_lyap;
  _mpc_params["CLF_W_CONTOUR"] = _w_qc_lyap;

  _mpc_params["USE_CBF"] = _use_cbf;
  _mpc_params["CBF_ALPHA_ABV"] = _cbf_alpha_abv;
  _mpc_params["CBF_ALPHA_BLW"] = _cbf_alpha_blw;
  _mpc_params["CBF_COLINEAR"] = _cbf_colinear;
  _mpc_params["CBF_PADDING"] = _cbf_padding;
  _mpc_params["CBF_DYNAMIC_ALPHA"] = _use_dynamic_alpha;

  _mpc_params["MAX_ANGA"] = _max_anga;
  _mpc_params["MAX_LINACC"] = _max_linacc;

  _mpc_params["TUBE_DEGREE"] = _tube_degree;
  _mpc_params["TUBE_SAMPLES"] = _tube_samples;
  _mpc_params["MAX_TUBE_WIDTH"] = _max_tube_width;

  _mpc_params["DEBUG"] = true;

  _mpc_core.load_params(_mpc_params);

  _mapSub = nh.subscribe("/grid_map", 1, &MPCCROS::mapcb, this);
  _odomSub = nh.subscribe("/odometry/filtered", 1, &MPCCROS::odomcb, this);
  _trajSub =
      nh.subscribe("/reference_trajectory", 1, &MPCCROS::trajectorycb, this);

  _timer = nh.createTimer(ros::Duration(_dt), &MPCCROS::mpcc_ctrl_loop, this);

  _startPub = nh.advertise<std_msgs::Float64>("/progress", 10);
  _pathPub = nh.advertise<nav_msgs::Path>("/spline_path", 10);
  _velPub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
  _trajPub = nh.advertise<nav_msgs::Path>("/mpc_prediction", 10);
  _solveTimePub = nh.advertise<std_msgs::Float64>("/mpc_solve_time", 0);
  _goalReachedPub = nh.advertise<std_msgs::Bool>("/mpc_goal_reached", 10);
  _pointPub = nh.advertise<geometry_msgs::PointStamped>("traj_point", 0);
  _refVizPub = nh.advertise<visualization_msgs::Marker>("/mpc_reference", 0);
  _tubeVizPub = nh.advertise<visualization_msgs::MarkerArray>("/tube_viz", 0);
  _horizonPub =
      nh.advertise<trajectory_msgs::JointTrajectory>("/mpc_horizon", 0);
  _refPub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>(
      "/current_reference", 10);

  timer_thread = std::thread(&MPCCROS::publishCmd, this);

  /*if (_is_eval) {*/
  /*  ROS_WARN("******************");*/
  /*  ROS_WARN("LOGGING IS ENABLED");*/
  /*  ROS_WARN("******************");*/
  /*}*/

  /*if (_use_cbf && (_is_logging || _is_eval)) {*/
  /*  std::unordered_map<std::string, double> logger_params;*/
  /*  logger_params["MIN_ALPHA"] = _min_alpha;*/
  /*  logger_params["MAX_ALPHA"] = _max_alpha;*/
  /*  logger_params["MIN_ALPHA_DOT"] = _min_alpha_dot;*/
  /*  logger_params["MAX_ALPHA_DOT"] = _max_alpha_dot;*/
  /*  logger_params["MIN_H_VAL"] = _min_h_val;*/
  /*  logger_params["MAX_H_VAL"] = _max_h_val;*/
  /*  logger_params["MAX_OBS_DIST"] = _max_tube_width;*/
  /*  logger_params["TASK_ID"] = _task_id;*/
  /*  logger_params["NUM_SAMPLES"] = _num_samples;*/
  /*  logger_params["MAX_PATH_LENGTH"] = _max_path_length;*/
  /**/
  /*  _logger =*/
  /*      std::make_unique<logger::RLLogger>(nh, logger_params, _is_logging);*/
  /**/
  /*} else if (!_use_cbf) {*/
  /*  _cbf_alpha_abv = kMAX_ALPHA;*/
  /*  _cbf_alpha_blw = kMAX_ALPHA;*/
  /*}*/

  // num coeffs is tube_W_ANGVELdegree + 1
  /*_tube_degree += 1;*/
}

MPCCROS::~MPCCROS() {
  if (timer_thread.joinable())
    timer_thread.join();
}

void MPCCROS::visualizeTubes() {
  using Side = mpcc::types::Corridor::Side;
  /*const mpcc::types::Trajectory& ref = _mpc_core->get_trajectory();*/
  /*double len_start                   = ref.get_closest_s(_odom.head(2));*/
  /*mpcc::types::Corridor corridor = _mpc_core->get_corridor(len_start);*/
  mpcc::types::Corridor corridor = _mpc_core.get_corridor(_odom.head(2));
  const mpcc::types::Trajectory &ref = corridor.get_trajectory();

  double len_start = ref.get_closest_s(_odom.head(2));
  ROS_INFO("CURRENT S: %.2f", len_start);

  /*double max_view_horizon = 4.0;*/
  /*double true_ref_len     = ref.get_arclen();*/
  /*double horizon          = true_ref_len;  //2 * _max_linvel * _dt *
   * _mpc_steps;*/
  double horizon = ref.get_arclen();

  /*if (len_start > true_ref_len)*/
  /*  return;*/
  /**/
  /*if (len_start + horizon > true_ref_len)*/
  /*  horizon = true_ref_len - len_start;*/

  /*horizon = std::min(horizon, max_view_horizon);*/

  visualization_msgs::Marker tubemsg_a;
  tubemsg_a.header.frame_id = _frame_id;
  tubemsg_a.header.stamp = ros::Time::now();
  tubemsg_a.ns = "tube_above";
  tubemsg_a.id = 87;
  tubemsg_a.action = visualization_msgs::Marker::ADD;
  tubemsg_a.type = visualization_msgs::Marker::LINE_STRIP;
  tubemsg_a.scale.x = .075;
  tubemsg_a.pose.orientation.w = 1;

  visualization_msgs::Marker tubemsg_b = tubemsg_a;
  tubemsg_b.header = tubemsg_a.header;
  tubemsg_b.ns = "tube_below";
  tubemsg_b.id = 88;

  // if horizon is that small, too small to visualize anyway
  if (horizon < .05)
    return;

  tubemsg_a.points.reserve(2 * (horizon / .05));
  tubemsg_b.points.reserve(2 * (horizon / .05));
  tubemsg_a.colors.reserve(2 * (horizon / .05));
  tubemsg_b.colors.reserve(2 * (horizon / .05));

  for (double s = 0; s < horizon; s += .05) {
    mpcc::types::Corridor::Sample corr_sample = corridor.get_at(s);

    geometry_msgs::Point &pt_a = tubemsg_a.points.emplace_back();
    pt_a.x = corr_sample.above(0);
    pt_a.y = corr_sample.above(1);
    pt_a.z = 1.0;

    geometry_msgs::Point &pt_b = tubemsg_b.points.emplace_back();
    pt_b.x = corr_sample.below(0);
    pt_b.y = corr_sample.below(1);
    pt_b.z = 1.0;

    // convenience for setting colors
    std_msgs::ColorRGBA color_msg_abv;
    color_msg_abv.r = 192. / 255.;
    color_msg_abv.g = 0.0;
    color_msg_abv.b = 0.0;
    color_msg_abv.a = 1.0;

    std_msgs::ColorRGBA color_msg_blw;
    color_msg_blw.r = 251. / 255.;
    color_msg_blw.g = 133. / 255.;
    color_msg_blw.b = 0.0;
    color_msg_blw.a = 1.0;

    tubemsg_a.colors.push_back(color_msg_abv);
    tubemsg_b.colors.push_back(color_msg_blw);
  }

  visualization_msgs::MarkerArray tube_ma;
  tube_ma.markers.reserve(2);
  tube_ma.markers.push_back(std::move(tubemsg_a));
  tube_ma.markers.push_back(std::move(tubemsg_b));

  _tubeVizPub.publish(tube_ma);
}

void MPCCROS::visualizeTraj() {
  visualization_msgs::Marker traj;
  traj.header.frame_id = _frame_id;
  traj.header.stamp = ros::Time::now();
  traj.ns = "mpc_reference";
  traj.id = 117;
  traj.action = visualization_msgs::Marker::ADD;
  traj.type = visualization_msgs::Marker::LINE_STRIP;
  traj.scale.x = .075;
  traj.pose.orientation.w = 1;

  const auto &reference = _mpc_core.get_trajectory();
  double true_ref_len = reference.get_arclen();
  for (double s = 0; s < true_ref_len; s += .05) {
    Eigen::Vector2d point = reference(s);

    geometry_msgs::Point &pt_a = traj.points.emplace_back();
    pt_a.x = point(0);
    pt_a.y = point(1);
    pt_a.z = 1.0;

    std_msgs::ColorRGBA color_msg;
    color_msg.r = 0;
    color_msg.g = 0.0;
    color_msg.b = 192. / 255.;
    color_msg.a = 1.0;

    traj.colors.push_back(color_msg);
  }

  _refVizPub.publish(traj);
}

void MPCCROS::publishCmd() {
  constexpr double pub_vel_loop_rate_hz = 50;
  const std::chrono::milliseconds pub_loop_period(
      static_cast<int>(1000.0 / pub_vel_loop_rate_hz));

  while (ros::ok()) {
    if (_trajectory.points.size() > 0)
      _velPub.publish(_vel_msg);

    // _velPub.publish(_vel_msg);

    std::this_thread::sleep_for(pub_loop_period);
  }
}

void MPCCROS::odomcb(const nav_msgs::Odometry::ConstPtr &msg) {
  tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                   msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  _odom = Eigen::VectorXd(3);

  _odom(0) = msg->pose.pose.position.x;
  _odom(1) = msg->pose.pose.position.y;
  _odom(2) = yaw;

  _mpc_core.set_odom(_odom);

  if (!_is_init) {
    _is_init = true;
    ROS_INFO("tracker initialized");
  }
}

/**********************************************************************
 * Function: MPCCROS::trajectorycb(const
 *trajectory_msgs::JointTrajectory::ConstPtr& msg) Description: Callback for
 *trajectory message Parameters:
 * @param msg: trajectory_msgs::JointTrajectory::ConstPtr
 * Returns:
 * N/A
 * Notes:
 * This function sets the reference trajectory for the MPC controller
 * Since the ACADOS MPC requires a hard coded trajectory size, the
 * trajectory is extended if it is less than the required size
 **********************************************************************/
void MPCCROS::trajectorycb(
    const trajectory_msgs::JointTrajectory::ConstPtr &msg) {
  ROS_INFO("Trajectory received!");
  _trajectory = *msg;

  if (msg->points.size() == 0) {
    ROS_WARN("Trajectory is empty, stopping!");
    _vel_msg.linear.x = 0;
    _vel_msg.angular.z = 0;
    return;
  }

  int N = msg->points.size();

  Eigen::VectorXd ss(N), xs(N), ys(N);
  for (int i = 0; i < N; ++i) {
    xs[i] = msg->points[i].positions[0];
    ys[i] = msg->points[i].positions[1];
    ss[i] = msg->points[i].time_from_start.toSec();
  }

  _mpc_core.set_trajectory(xs, ys, ss);
  _is_traj_set = true;

  visualizeTraj();

  ROS_INFO("**********************************************************");
  ROS_INFO("MPC received trajectory! Length: %.2f",
           _mpc_core.get_trajectory().get_arclen());
  ROS_INFO("**********************************************************");
}

bool MPCCROS::can_execute() { return _is_init && _is_traj_set; }

void MPCCROS::mpcc_ctrl_loop(const ros::TimerEvent &event) {
  if (!can_execute())
    return;

  const mpcc::types::Trajectory &trajectory = _mpc_core.get_trajectory();
  double true_ref_len = trajectory.get_arclen();
  double len_start = trajectory.get_closest_s(_odom.head(2));

  ROS_INFO("len_start is: %.2f / %.2f", len_start, true_ref_len);

  std_msgs::Float64 start_msg;
  start_msg.data = len_start / true_ref_len;
  _startPub.publish(start_msg);

  if (len_start > true_ref_len - 0.25) {
    ROS_INFO("Reached end of traj %.2f / %.2f", len_start, true_ref_len);
    _vel_msg.angular.z = 0;
    if (_mpc_input_type == mpcc::MPCType::kUnicycle)
      _vel_msg.linear.x = 0;
    else if (_mpc_input_type == mpcc::MPCType::kDoubleIntegrator) {
      _vel_msg.linear.x = 0;
      _vel_msg.linear.y = 0;
    }

    _trajectory.points.clear();

    return;
  }

  ros::Time now = ros::Time::now();

  Eigen::VectorXd state(4);
  if (_mpc_input_type == mpcc::MPCType::kUnicycle)
    state << _odom(0), _odom(1), _odom(2), _vel_msg.linear.x;
  else if (_mpc_input_type == mpcc::MPCType::kDoubleIntegrator)
    state << _odom(0), _odom(1), _vel_msg.linear.x, _vel_msg.linear.y;
  else {
    ROS_ERROR("Unknown MPC input type: %d",
              static_cast<unsigned int>(_mpc_input_type));
    return;
  }

  // before solve get update alpha values if dynamic alpha is enabled
  /*if (_logger) {*/
  /*  _logger->request_alpha(*_mpc_core);*/
  /*}*/

  ROS_INFO("calling mpc core solve");
  std::array<double, 2> input = _mpc_core.solve(state);

  if (_mpc_input_type == mpcc::MPCType::kUnicycle) {
    _vel_msg.linear.x = input[0];
    _vel_msg.angular.z = input[1];
  } else if (_mpc_input_type == mpcc::MPCType::kDoubleIntegrator) {
    _vel_msg.linear.x = input[0];
    _vel_msg.linear.y = input[1];
  } else {
    ROS_ERROR("Unknown MPC input type: %d",
              static_cast<unsigned int>(_mpc_input_type));

    return;
  }

  // log data back to db if logging enabled
  /*if (_is_logging || _is_eval)*/
  /*_mpc_core.get_cbf_data(0);*/

  ROS_WARN("runtime: %.3f", (ros::Time::now() - now).toSec());

  publishReference();

  publishMPCTrajectory();
  visualizeTubes();

  geometry_msgs::PointStamped pt;
  pt.header.frame_id = _frame_id;
  pt.point.z = .1;

  double s = trajectory.get_closest_s(_odom.head(2));
  Eigen::Vector2d ref_pt = trajectory(s);

  pt.header.stamp = ros::Time::now();
  pt.point.x = ref_pt(0);
  pt.point.y = ref_pt(1);

  _pointPub.publish(pt);
}

void MPCCROS::publishReference() {
  if (_trajectory.points.size() == 0)
    return;

  nav_msgs::Path msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = _frame_id;
  msg.poses.reserve(_trajectory.points.size());

  bool published = false;
  for (const trajectory_msgs::JointTrajectoryPoint &pt : _trajectory.points) {
    if (!published) {
      published = true;
      _refPub.publish(pt);
    }

    geometry_msgs::PoseStamped &pose = msg.poses.emplace_back();
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = _frame_id;
    pose.pose.position.x = pt.positions[0];
    pose.pose.position.y = pt.positions[1];
    pose.pose.position.z = 0;
    pose.pose.orientation.x = 0;
    pose.pose.orientation.y = 0;
    pose.pose.orientation.z = 0;
    pose.pose.orientation.w = 1;
  }

  _pathPub.publish(msg);
}

void MPCCROS::publishMPCTrajectory() {
  mpcc::MPCCore::AnyHorizon horizon = _mpc_core.get_horizon();

  size_t horizon_steps =
      std::visit([](const auto &arg) { return arg.length; }, horizon);

  if (horizon_steps == 0) {
    return;
  }

  geometry_msgs::PoseStamped goal;
  goal.header.stamp = ros::Time::now();
  goal.header.frame_id = _frame_id;
  goal.pose.position.x = _x_goal;
  goal.pose.position.y = _y_goal;
  goal.pose.orientation.w = 1;

  nav_msgs::Path pathMsg;
  pathMsg.header.frame_id = _frame_id;
  pathMsg.header.stamp = ros::Time::now();

  const mpcc::types::Trajectory &reference = _mpc_core.get_trajectory();
  double true_ref_len = reference.get_arclen();

  for (int step = 0; step < horizon_steps; ++step) {
    // don't visualize mpc horizon past end of reference trajectory
    // eventually fix this so that MPCC ros does not need to know this much
    // info about lower level class structure
    double s = std::visit(
        [&](const auto &arg) { return arg.get_arclen_at_step(step); }, horizon);

    if (s > true_ref_len) {
      break;
    }

    const Eigen::VectorXd &pos = std::visit(
        [&](const auto &arg) { return arg.get_pos_at_step(step); }, horizon);

    geometry_msgs::PoseStamped tmp;
    tmp.header = pathMsg.header;
    tmp.pose.position.x = pos(0);
    tmp.pose.position.y = pos(1);
    tmp.pose.position.z = .1;
    tmp.pose.orientation.w = 1;
    pathMsg.poses.push_back(tmp);
  }

  _trajPub.publish(pathMsg);

  trajectory_msgs::JointTrajectory traj;
  traj.header.stamp = ros::Time::now();
  traj.header.frame_id = _frame_id;

  for (int step = 0; step < horizon_steps; ++step) {

    const Eigen::VectorXd &pos = std::visit(
        [&](const auto &arg) { return arg.get_pos_at_step(step); }, horizon);

    const Eigen::VectorXd &vel = std::visit(
        [&](const auto &arg) { return arg.get_vel_at_step(step); }, horizon);

    const Eigen::VectorXd &acc = std::visit(
        [&](const auto &arg) { return arg.get_vel_at_step(step); }, horizon);

    // manually compute jerk in x and y directions from acceleration
    /*double jerk_x = 0;*/
    /*double jerk_y = 0;*/
    Eigen::VectorXd jerk;
    if (step < horizon_steps - 1) {
      const Eigen::VectorXd &next_acc = std::visit(
          [&](const auto &arg) { return arg.get_vel_at_step(step + 1); },
          horizon);
      jerk = (next_acc - acc) / _dt;
    } else {
      jerk = Eigen::VectorXd::Zero(vel.size());
    }

    trajectory_msgs::JointTrajectoryPoint pt;
    pt.time_from_start = ros::Duration(step * _dt);
    pt.positions = {pos(0), pos(1), 0};
    pt.velocities = {vel(0), vel(1), 0};
    pt.accelerations = {acc(0), acc(1), 0};
    pt.effort = {jerk(0), jerk(1), 0};

    traj.points.push_back(pt);
  }

  _horizonPub.publish(traj);
}
