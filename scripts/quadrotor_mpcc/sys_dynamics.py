import numpy as np
import casadi as ca
from acados_template import AcadosModel
from common import *


def create_interp(name, knots, coeffs):
    s = ca.MX.sym("s")
    interp = ca.interpolant(name, "bspline", [knots.tolist()])
    return ca.Function(name, [s, coeffs], [interp(s, coeffs)])


class SysDyn:

    def __init__(self, tube_degree):
        self.tube_degree = tube_degree

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

        vx_coeff = ca.MX.sym("vx_coeffs", n_knots)
        vxr_func = create_interp("interp_vx", arc_len_knots, vx_coeff)
        vy_coeff = ca.MX.sym("vy_coeffs", n_knots)
        vyr_func = create_interp("interp_vy", arc_len_knots, vy_coeff)
        vz_coeff = ca.MX.sym("vz_coeffs", n_knots)
        vzr_func = create_interp("interp_vz", arc_len_knots, vz_coeff)

        e_axis_a_coeff = ca.MX.sym("e_axis_a_coeff", self.tube_degree + 1)
        e_axis_b_coeff = ca.MX.sym("e_axis_b_coeff", self.tube_degree + 1)
        e_offset_a_coeff = ca.MX.sym("e_offset_a_coeff", self.tube_degree + 1)
        e_offset_b_coeff = ca.MX.sym("e_offset_b_coeff", self.tube_degree + 1)


        s_norm = s / L_path
        xr = xr_func(s_norm, x_coeff)
        yr = yr_func(s_norm, y_coeff)
        zr = zr_func(s_norm, z_coeff)
        pr = ca.vertcat(xr, yr, zr)

        n_terms = self.tube_degree + 1
        T = casadi_chebyshev_basis(s_norm, self.tube_degree)
        e_axis_a = sum(e_axis_a_coeff[i] * T[i] for i in range(n_terms))
        e_axis_b = sum(e_axis_b_coeff[i] * T[i] for i in range(n_terms))
        e_offset_a = sum(e_offset_a_coeff[i] * T[i] for i in range(n_terms))
        e_offset_b = sum(e_offset_b_coeff[i] * T[i] for i in range(n_terms))
        # e_axis_a = 0
        # e_axis_b = 0
        # e_offset_a = 0
        # e_offset_b = 0
        # for i in range(self.tube_degree + 1):
        #     # chebyshev basis...
        #     basis_i = ca.cos(i * ca.acos(2 * s_norm - 1))
        #     e_axis_a = e_axis_a + e_axis_a_coeff[i] * basis_i
        #     e_axis_b = e_axis_b + e_axis_b_coeff[i] * basis_i
        #     e_offset_a = e_offset_a + e_offset_a_coeff[i] * basis_i
        #     e_offset_b = e_offset_b + e_offset_b_coeff[i] * basis_i


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
        # cbf = radius**2 - ca.dot(e_perp, e_perp) - 0.05 * alignment
        err_e1 = ca.dot(e_tot, e1)
        err_e2 = ca.dot(e_tot, e2)
        # err_rmf = ca.vertcat(err_e1, err_e2)

        E = ca.vertcat(
            ca.horzcat(e_axis_a, 0),
            ca.horzcat(0, e_axis_b)
        )

        # ellipse_center = ca.vertcat(e_offset_a, e_offset_b)


        # cbf = 1 - (ellipse_center - err_rmf).T @ E @ (ellipse_center - err_rmf)
        cx = -e_offset_a / (2 * e_axis_a)
        cy = -e_offset_b / (2 * e_axis_b)
        w1_shifted = err_e1 - cx
        w2_shifted = err_e2 - cy

        cbf = 1 - (e_axis_a * w1_shifted**2 + e_axis_b * w2_shifted**2)
        # cbf = 1 - (e_axis_a * err_e1**2 + 
        #            e_axis_b * err_e2**2 + 
        #            e_offset_a * err_e1 + 
        #            e_offset_b * err_e2)

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
        f_danger = 0.5 * ca.dot(e_perp, v) + 0.1 * ca.dot(e_perp, z_b)

        # h_pos = radius**2 - ca.dot(e_perp, e_perp)
        # cbf = h_pos * ca.exp(-f_danger)
        # cbf = h_pos / (1.0 + f_danger**2)

        grad_h = ca.jacobian(cbf, x)
        Lfh = grad_h @ F
        Lgh = grad_h @ G
        grad_Lfh = ca.jacobian(Lfh, x)
        Lf2h = grad_Lfh @ F
        LgLfh = grad_Lfh @ G
        hddot = Lf2h + LgLfh @ u

        # a_des = (thrust / mq) * Rq[:, 2] - ca.vertcat(0, 0, 9.81)

        # Lfh = ca.jacobian(cbf, pos) @ v
        # Lf2h = ca.jacobian(Lfh, pos) @ v + ca.jacobian(Lfh, v) @ a_des
        # LgLfh_a = ca.jacobian(Lfh, v)
        # hddot = Lf2h + LgLfh_a @ a_des

        # cbf_cons = Lf2h + LgLfh_a @ a_des + 1 * Lfh + 1 * cbf

        alpha0 = ca.MX.sym("alpha0")
        alpha1 = ca.MX.sym("alpha1")
        cbf_cons = Lf2h + LgLfh @ u + alpha1 * Lfh + alpha0 * cbf


        # cbf_cons = hddot + 10 * Lfh + 0.02 * cbf
        # cbf_cons = Lfh + Lgh @ u + alpha0 * cbf

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
            e_axis_a_coeff,
            e_axis_b_coeff,
            e_offset_a_coeff,
            e_offset_b_coeff,
            L_path,
            Q_c,
            Q_l,
            Q_t,
            Q_w,
            Q_sdd,
            Q_s,
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
                e_axis_a_coeff,
                e_axis_b_coeff,
                e_offset_a_coeff,
                e_offset_b_coeff,
                L_path,
                Q_c,
                Q_l,
                Q_t,
                Q_w,
                Q_sdd,
                Q_s,
                alpha0,
                alpha1,
                x,
                u,
            ],
            [hddot, Lfh, cbf],
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
                "e_axis_a_coeff",
                "e_axis_b_coeff",
                "e_offset_a_coeff",
                "e_offset_b_coeff",
                "L",
                "Q_c",
                "Q_l",
                "Q_t",
                "Q_w",
                "Q_sdd",
                "Q_s",
                "alpha0",
                "alpha1",
                "x_val",
                "u",
            ],
            ["hddot", "Lfh", "cbf"],
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
