#ifndef QUADROTOR_MPCC_MPCC_ROS_H
#define QUADROTOR_MPCC_MPCC_ROS_H

#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <mpcc/ros/logger.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_srvs/Empty.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <string>
#include <thread>

#include <mpcc/common/mpcc_core.h>

class MPCCROS {
 public:
  MPCCROS(ros::NodeHandle& nh);
  ~MPCCROS();

 private:
  void publishMPCTrajectory();
  /**********************************************************************
     * Function: MPCCROS::publishMPCTrajectory()
     * Description: Publishes the MPC prediction horizon
     * Parameters:
     * N/A
     * Returns:
     * N/A
     * Notes:
     * This function outputs the trajectory in JointTrajectory form so
     * trajectory generators can determine initial pos, vel, acc, etc.
     * for initial seeding.
     **********************************************************************/

  void publishReference();
  /**********************************************************************
     * Function: MPCCROS::publishReference()
     * Description: Publishes the reference trajectory
     * Parameters:
     * N/A
     * Returns:
     * N/A
     **********************************************************************/

  void mpcc_ctrl_loop(const ros::TimerEvent& event);
  /**********************************************************************
     * Function: MPCCROS::mpcc_ctrl_loop()
     * Description: Main control loop for MPC controller
     * Parameters:
     * Returns:
     * N/A
     * Notes:
     * Main control loop for the MPC, responsible for generating CBF tubes
     * and calling the MPC solver. Also sets up the virtual state s_dot
     **********************************************************************/

  /**********************************************************************
     * Callbacks for CBF alpha parameter, map, goal (not implemented
     * currently), odometry, and trajectory
     **********************************************************************/
  void odomcb(const nav_msgs::Odometry::ConstPtr& msg);
  void trajectorycb(const trajectory_msgs::JointTrajectory::ConstPtr& msg);

  void publishCmd();
  /**********************************************************************
     * Function: MPCCROS::publishVel()
     * Description: Publishes velocity command
     * Parameters:
     * Returns:
     * N/A
     * Notes:
     * Some vehicles require very high velocity publish rates (BD SPOT),
     * so the publishing of velocity is done in this separate thread at
     * a much higher frequency than the control loop.
     **********************************************************************/

  void visualizeTraj();

  bool can_execute();

  /************************
     * Class variables
     ************************/

  mpcc::MPCCore _mpc_core;
  /**********************************************************************
     * In previous projects this has been the wrapper that can switch
     * between different MPC class implementations, but in this project only
     * one is currently implemented (the MPCC). Will eventually add more.
     **********************************************************************/
  /*std::unique_ptr<logger::RLLogger> _logger;*/

  ros::Subscriber _trajSub;
  ros::Subscriber _trajNoResetSub;
  ros::Subscriber _obsSub;
  ros::Subscriber _alphaSub;
  ros::Subscriber _odomSub;
  ros::Subscriber _collisionSub;
  ros::Subscriber _mapSub;

  ros::Publisher _velPub;
  ros::Publisher _trajPub;
  ros::Publisher _pathPub;
  ros::Publisher _pointPub;
  ros::Publisher _odomPub;
  ros::Publisher _refPub;
  ros::Publisher _goalReachedPub;
  ros::Publisher _horizonPub;
  ros::Publisher _solveTimePub;
  ros::Publisher _donePub;
  ros::Publisher _loggingPub;
  ros::Publisher _tubeVizPub;
  ros::Publisher _refVizPub;
  ros::Publisher _startPub;

  ros::ServiceClient _sac_srv;

  ros::NodeHandle _nh;

  ros::Timer _timer, _velPubTimer;

  Eigen::VectorXd _odom;

  trajectory_msgs::JointTrajectory _trajectory;

  costmap_2d::Costmap2DROS* _local_costmap;

  std::vector<Eigen::Vector3d> poses;
  std::vector<double> mpc_results;

  std::map<std::string, double> _mpc_params;

  double _mpc_steps, _w_vel, _w_angvel, _w_linvel, _w_angvel_d, _w_linvel_d,
      _w_etheta, _max_angvel, _max_linvel, _bound_value, _x_goal, _y_goal,
      _theta_goal, _tol, _max_linacc, _max_anga, _w_cte, _w_pos, _w_qc, _w_ql,
      _w_q_speed;

  double _cbf_alpha_abv, _cbf_alpha_blw, _cbf_colinear, _cbf_padding;

  double _prop_gain, _prop_angle_thresh;

  double _clf_gamma;
  double _w_ql_lyap;
  double _w_qc_lyap;

  double _min_alpha;
  double _max_alpha;
  double _min_alpha_dot;
  double _max_alpha_dot;
  double _min_h_val;
  double _max_h_val;

  double _ref_len;
  double _true_ref_len;
  double _mpc_ref_len_sz;
  double _max_tube_width;

  double _dt, _curr_vel, _curr_ang_vel, _vel_pub_freq;
  bool _is_init, _is_goal, _teleop, _estop, _is_at_goal, _use_cbf,
      _use_dynamic_alpha;

  bool _is_traj_set{false};
  bool _is_logging;
  bool _is_eval;

  int _task_id;
  int _num_samples;
  int _tube_degree;
  int _tube_samples;
  int _max_path_length;
  int _mpc_ref_samples;

  grid_map::GridMap _grid_map;

  Eigen::MatrixX4d _poly;
  geometry_msgs::Twist _vel_msg;

  Eigen::VectorXd _prev_rl_state;
  Eigen::VectorXd _curr_rl_state;

  std::string _frame_id;
  std::string _logging_table_name;
  std::string _logging_topic_name;

  mpcc::MPCType _mpc_input_type;

  std::thread timer_thread;

  static constexpr double kMAX_ALPHA = 100.f;
};

#endif
