import numpy as np
import casadi as ca
from acados_template import AcadosModel
from common import *


def create_interp(name, knots, coeffs):
    s = ca.MX.sym("s")
    interp = ca.interpolant(name, "bspline", [knots.tolist()])
    return ca.Function(name, [s, coeffs], [interp(s, coeffs)])


class SysDyn:

    def __init__(self):
        pass

    def setup(self):

        model_name = "quadrotor_mpcc"

        # dynamics
        px = ca.MX.sym("px")
        py = ca.MX.sym("py")
        pz = ca.MX.sym("pz")
        pos = ca.vertcat(px, py, pz)

        vx = ca.MX.sym("vx")
        vy = ca.MX.sym("vy")
        vz = ca.MX.sym("vz")
        v = ca.vertcat(vx, vy, vz)

        q1 = ca.MX.sym("q1")
        q2 = ca.MX.sym("q2")
        q3 = ca.MX.sym("q3")
        q4 = ca.MX.sym("q4")
        q = ca.vertcat(q1, q2, q3, q4)

        s = ca.MX.sym("s")
        s_dot = ca.MX.sym("s_dot")

        x = ca.vertcat(pos, v, q, s, s_dot)

        # inputs
        wr = ca.MX.sym("wr")
        wp = ca.MX.sym("wp")
        wy = ca.MX.sym("wy")
        omg = ca.vertcat(wr, wp, wy)

        thrust = ca.MX.sym("thrust")
        s_ddot = ca.MX.sym("s_ddot")
        u = ca.vertcat(thrust, omg, s_ddot)

        q1_dot = (-q2 * wr - q3 * wp - q4 * wy) / 2
        q2_dot = (q1 * wr + q3 * wy - q4 * wp) / 2
        q3_dot = (q1 * wp - q2 * wy + q4 * wr) / 2
        q4_dot = (q1 * wy + q2 * wp - q3 * wr) / 2
        qdot = ca.vertcat(q1_dot, q2_dot, q3_dot, q4_dot)

        Rq = ca.vertcat(
            ca.horzcat(
                1 - 2 * (q3**2 + q4**2),
                2 * (q2 * q3 - q1 * q4),
                2 * (q2 * q4 + q1 * q3),
            ),
            ca.horzcat(
                2 * (q2 * q3 + q1 * q4),
                1 - 2 * (q2**2 + q4**2),
                2 * (q3 * q4 - q1 * q2),
            ),
            ca.horzcat(
                2 * (q2 * q4 - q1 * q3),
                2 * (q3 * q4 + q1 * q2),
                1 - 2 * (q2**2 + q3**2),
            ),
        )

        thrust_vector = ca.vertcat(0, 0, thrust)
        gravity = ca.vertcat(0, 0, g0)
        vdot = -gravity + (1 / mq) * Rq @ thrust_vector

        f_expl = ca.vertcat(v, vdot, qdot, s_dot, s_ddot)

        # tracking error

        L_path = ca.MX.sym("L_path", 1)

        arc_len_knots = np.linspace(0, 1, n_knots)

        # xspl = ca.MX.sym("xspl", 1, 1)
        # yspl = ca.MX.sym("yspl", 1, 1)
        # zspl = ca.MX.sym("zspl", 1, 1)

        x_coeff = ca.MX.sym("x_coeffs", n_knots)
        xr_func = create_interp("xr", arc_len_knots, x_coeff)
        y_coeff = ca.MX.sym("y_coeffs", n_knots)
        yr_func = create_interp("yr", arc_len_knots, y_coeff)
        z_coeff = ca.MX.sym("z_coeffs", n_knots)
        zr_func = create_interp("zr", arc_len_knots, z_coeff)

        e1x_coeff = ca.MX.sym("e1x_coeffs", n_knots)
        e1x_func = create_interp("interp_e1x", arc_len_knots, e1x_coeff)
        e1y_coeff = ca.MX.sym("e1y_coeffs", n_knots)
        e1y_func = create_interp("interp_e1y", arc_len_knots, e1y_coeff)
        e1z_coeff = ca.MX.sym("e1z_coeffs", n_knots)
        e1z_func = create_interp("interp_e1z", arc_len_knots, e1z_coeff)

        e2x_coeff = ca.MX.sym("e2x_coeffs", n_knots)
        e2x_func = create_interp("interp_e2x", arc_len_knots, e2x_coeff)
        e2y_coeff = ca.MX.sym("e2y_coeffs", n_knots)
        e2y_func = create_interp("interp_e2y", arc_len_knots, e2y_coeff)
        e2z_coeff = ca.MX.sym("e2z_coeffs", n_knots)
        e2z_func = create_interp("interp_e2z", arc_len_knots, e2z_coeff)

        vx_coeff = ca.MX.sym("vx_coeffs", n_knots)
        vxr_func = create_interp("interp_vx", arc_len_knots, vx_coeff)
        vy_coeff = ca.MX.sym("vy_coeffs", n_knots)
        vyr_func = create_interp("interp_vy", arc_len_knots, vy_coeff)
        vz_coeff = ca.MX.sym("vz_coeffs", n_knots)
        vzr_func = create_interp("interp_vz", arc_len_knots, vz_coeff)

        s_norm = s / L_path
        xr = xr_func(s_norm, x_coeff)
        yr = yr_func(s_norm, y_coeff)
        zr = zr_func(s_norm, z_coeff)
        pr = ca.vertcat(xr, yr, zr)

        xr_dot = vxr_func(s_norm, vx_coeff)
        yr_dot = vyr_func(s_norm, vy_coeff)
        zr_dot = vzr_func(s_norm, vz_coeff)
        tr_raw = ca.vertcat(xr_dot, yr_dot, zr_dot)
        tr = tr_raw / (ca.norm_2(tr_raw) + 1e-8)

        # cost
        Q_c = ca.MX.sym("Q_c")
        Q_t = ca.MX.sym("Q_t")
        Q_l = ca.MX.sym("Q_l")
        Q_s = ca.MX.sym("Q_s")
        Q_w = ca.MX.sym("Q_w")
        Q_sdd = ca.MX.sym("Q_sdd")

        e_tot = pos - pr
        e_l = ca.dot(tr, e_tot)
        e_l_vec = e_l * tr
        e_c = e_tot - e_l_vec

        cost_expr = (
            Q_c * ca.dot(e_c, e_c)
            + Q_l * e_l**2
            + Q_t * thrust**2
            + Q_w * ca.dot(omg, omg)
            + Q_sdd * s_ddot**2
            - Q_s * s_dot
        )

        cost_expr_e = Q_c * ca.dot(e_c, e_c) + Q_l * e_l**2 - Q_s * s_dot

        # control barrier function
        e1x = e1x_func(s_norm, e1x_coeff)
        e1y = e1y_func(s_norm, e1y_coeff)
        e1z = e1z_func(s_norm, e1z_coeff)
        e1_raw = ca.vertcat(e1x, e1y, e1z)
        e1 = e1_raw / (ca.norm_2(e1_raw) + 1e-8)

        # e2x = e2x_func(s_norm, e2x_coeff)
        # e2y = e2y_func(s_norm, e2y_coeff)
        # e2z = e2z_func(s_norm, e2z_coeff)
        # e2_raw = ca.vertcat(e2x, e2y, e2z)
        # e2 = e2_raw / (ca.norm_2(e2_raw) + 1e-8)
        e2 = ca.cross(tr, e1)

        b1 = ca.dot(e1, e_tot)
        b2 = ca.dot(e2, e_tot)

        # CBF
        F = ca.vertcat(v, -gravity, ca.MX.zeros(4, 1), s_dot, 0)
        G = ca.MX.zeros(x.size1(), u.size1())
        G[3:6, 0] = (1 / mq) * Rq[:, 2]
        G[6:10, 1:4] = 0.5 * ca.vertcat(
            ca.horzcat(-q2, -q3, -q4),
            ca.horzcat(q1, -q4, q3),
            ca.horzcat(q4, q1, -q2),
            ca.horzcat(-q3, q2, q1),
        )
        G[11, 4] = 1

        e_perp = e_tot - e_l_vec
        v_perp = v - ca.dot(v, tr) * tr

        alignment = ca.dot(e_perp, v)

        radius = 1.0
        # cbf = radius - dist_to_path
        # cbf = radius**2 - ca.dot(e_perp, e_perp) - 2 * .05 * ca.dot(e_perp, v_perp)
        cbf = radius**2 - ca.dot(e_perp, e_perp) - .05 * alignment

        # perp_dir = e_perp / (
        #     ca.norm_2(e_perp) + 1e-8
        # )  # Unit vector pointing away from path
        #
        # thrust_dir = Rq[:, 2]
        # thrust_perp = (
        #     thrust_dir - ca.dot(thrust_dir, tr) * tr
        # )  # Remove tangential component
        # thrust_toward_boundary = ca.dot(thrust_perp, perp_dir)
        #
        # v_perp = v - ca.dot(v, tr) * tr  # Remove tangential component
        # v_toward_boundary = ca.dot(v_perp, perp_dir)
        #
        # beta = 0.05
        # # p = thrust_toward_boundary  + beta * v_toward_boundary
        # p = beta * v_toward_boundary
        # p_clamped = ca.fmax(ca.fmin(p, 1.0), -1.0)
        # cbf = h * ca.exp(-p_clamped)

        # radius = 0.5
        # h = 1 - (b1 / radius) ** 2 - (b2 / radius) ** 2
        #
        # thrust_dir = Rq[:, 2]
        #
        # contour_mag = ca.sqrt(b1**2 + b2**2 + 1e-8)
        # boundary_dir_3d = (b1 * e1 + b2 * e2) / contour_mag
        #
        # thrust_toward_boundary = ca.dot(thrust_dir, boundary_dir_3d)
        #
        # v_toward_boundary = ca.dot(v, boundary_dir_3d)
        #
        # p = thrust_toward_boundary + 0.05 * v_toward_boundary
        # p_sat = ca.fmax(ca.fmin(p, 1.0), -1.0)
        # cbf = h * ca.exp(-p_sat)

        # v_e1 = ca.dot(e1, v)
        # v_e2 = ca.dot(e2, v)

        # contour_mag = ca.sqrt(b1**2 + b2**2 + 1e-8)
        # boundary_dir_e1 = b1 / contour_mag
        # boundary_dir_e2 = b2 / contour_mag
        #
        # v_toward_boundary = boundary_dir_e1 * v_e1 + boundary_dir_e2 * v_e2
        #
        # v_speed = ca.sqrt(ca.dot(v, v) + 1e-8)
        # p = v_toward_boundary + 0.05 * v_speed
        #
        # cbf = h * ca.exp(-p)

        z_b = Rq[:, 2]
        f_danger = .05 * ca.dot(e_perp, v) + .1 * ca.dot(e_perp, z_b)

        h_pos = radius**2 - ca.dot(e_perp, e_perp)
        cbf = h_pos * ca.exp(-f_danger)
        # cbf = h_pos / (1.0 + f_danger**2)

        grad_h = ca.jacobian(cbf, x)
        Lfh = grad_h @ F
        Lgh = grad_h @ G
        grad_Lfh = ca.jacobian(Lfh, x)
        Lf2h = grad_Lfh @ F
        LgLfh = grad_Lfh @ G

        alpha0 = ca.MX.sym("alpha0")
        alpha1 = ca.MX.sym("alpha1")

        hddot = Lf2h + LgLfh @ u
        # cbf_cons = hddot + alpha1 * Lfh + alpha0 * cbf
        cbf_cons = Lfh + Lgh@u + alpha0 * cbf

        # control lyap
        # tr_dot = ca.jacobian(tr, s) * s_dot
        # e_tot_dot = v - ca.jacobian(pr, s) * s_dot  # ṗ - ṗ_ref
        # e_l_dot = ca.dot(tr_dot, e_tot) + ca.dot(tr, e_tot_dot)
        #
        # lambda_lag = 1  # Tunable parameter
        # V_lag = 0.5 * (e_l + lambda_lag * e_l_dot) ** 2
        #
        # gamma_lag = 50  # Tunable parameter
        # grad_V_lag = ca.jacobian(V_lag, x)
        # V_lag_dot = grad_V_lag @ F + grad_V_lag @ G @ u
        #
        # clf_cons = V_lag_dot + gamma_lag * V_lag

        p = ca.vertcat(
            x_coeff,
            y_coeff,
            z_coeff,
            vx_coeff,
            vy_coeff,
            vz_coeff,
            e1x_coeff,
            e1y_coeff,
            e1z_coeff,
            e2x_coeff,
            e2y_coeff,
            e2z_coeff,
            Q_c,
            Q_l,
            Q_t,
            Q_w,
            Q_sdd,
            Q_s,
            L_path,
            alpha0,
            alpha1,
        )

        s_cons = s - L_path

        cbf_logic_func = ca.Function(
            "cbf_logic",
            [
                x_coeff,
                y_coeff,
                z_coeff,
                vx_coeff,
                vy_coeff,
                vz_coeff,
                e1x_coeff,
                e1y_coeff,
                e1z_coeff,
                e2x_coeff,
                e2y_coeff,
                e2z_coeff,
                L_path,
                x,
                u,
            ],
            [hddot, Lfh, cbf, Lgh],
            [
                "x_c",
                "y_c",
                "z_c",
                "vx_c",
                "vy_c",
                "vz_c",
                "e1x_c",
                "e1y_c",
                "e1z_c",
                "e2x_c",
                "e2y_c",
                "e2z_c",
                "L",
                "x_val",
                "u",
            ],
            ["hddot", "Lfh", "cbf", "LgLfh"],
        )

        model = AcadosModel()
        # model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p
        # model.con_h_expr_0 = ca.vertcat(clf_cons, cbf_cons, s_cons)
        # model.con_h_expr = ca.vertcat(clf_cons, cbf_cons, s_cons)

        model.con_h_expr_0 = ca.vertcat(cbf_cons, s_cons)
        model.con_h_expr = ca.vertcat(cbf_cons, s_cons)

        # model.con_h_expr_0 = ca.vertcat(s_cons)
        # model.con_h_expr = ca.vertcat(s_cons)
        model.con_h_expr_e = ca.vertcat(s_cons)

        model.cost_expr_ext_cost = cost_expr
        model.cost_expr_ext_cost_e = cost_expr_e
        # model.xdot = xdot
        model.name = model_name

        # store meta information
        model.x_labels = [
            "$px$ [m]",
            "$py$ [m]",
            "$pz$ [m]",
            "$vx$ [m/s]",
            "$vy$ [m/s]",
            "$vz$ [m/s]",
            "$ax$ [m/s2]",
            "$ay$ [m/s2]",
            "$az$ [m/s2]",
            "$s$ []",
            "$sdot$ []",
        ]
        model.u_labels = ["$jx$", "$jy$", "$jz$", "$sddot$"]
        model.t_label = "$t$ [s]"

        return model, cbf_logic_func
