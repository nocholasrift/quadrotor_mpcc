import numpy as np
import casadi as ca

from common import *
from acados_template import AcadosModel


def create_interp(name, knots, coeffs):
    s = ca.MX.sym("s")
    interp = ca.interpolant(name, "bspline", [knots.tolist()])
    return ca.Function(name, [s, coeffs], [interp(s, coeffs)])


class SysDyn:

    def __init__(self, tube_degree):
        self.tube_degree = tube_degree

    def setup(self):

        model_name = "double_integrator"

        # states
        px = ca.MX.sym("px")
        py = ca.MX.sym("py")
        pz = ca.MX.sym("pz")
        pos = ca.vertcat(px, py, pz)

        vx = ca.MX.sym("vx")
        vy = ca.MX.sym("vy")
        vz = ca.MX.sym("vz")
        v = ca.vertcat(vx, vy, vz)

        s = ca.MX.sym("s")
        s_dot = ca.MX.sym("s_dot")

        x = ca.vertcat(pos, v, s, s_dot)

        # inputs
        ax = ca.MX.sym("ax")
        ay = ca.MX.sym("ay")
        az = ca.MX.sym("az")
        a = ca.vertcat(ax, ay, az)

        s_ddot = ca.MX.sym("s_ddot")

        u = ca.vertcat(a, s_ddot)

        f_expl = ca.vertcat(v, a, s_dot, s_ddot)

        L_path = ca.MX.sym("L_path", 1)

        arc_len_knots = np.linspace(0, 1, n_knots)

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

        # e_axis_a_coeff = ca.MX.sym("e_axis_a_coeff", self.tube_degree + 1)
        # e_axis_b_coeff = ca.MX.sym("e_axis_b_coeff", self.tube_degree + 1)
        # e_offset_a_coeff = ca.MX.sym("e_offset_a_coeff", self.tube_degree + 1)
        # e_offset_b_coeff = ca.MX.sym("e_offset_b_coeff", self.tube_degree + 1)

        s_norm = s / L_path
        xr = xr_func(s_norm, x_coeff)
        yr = yr_func(s_norm, y_coeff)
        zr = zr_func(s_norm, z_coeff)
        pr = ca.vertcat(xr, yr, zr)

        n_terms = self.tube_degree + 1
        # T = casadi_chebyshev_basis(s_norm, self.tube_degree)
        # e_axis_a = sum(e_axis_a_coeff[i] * T[i] for i in range(n_terms))
        # e_axis_b = sum(e_axis_b_coeff[i] * T[i] for i in range(n_terms))
        # e_offset_a = sum(e_offset_a_coeff[i] * T[i] for i in range(n_terms))
        # e_offset_b = sum(e_offset_b_coeff[i] * T[i] for i in range(n_terms))

        xr_dot = vxr_func(s_norm, vx_coeff)
        yr_dot = vyr_func(s_norm, vy_coeff)
        zr_dot = vzr_func(s_norm, vz_coeff)
        tr_raw = ca.vertcat(xr_dot, yr_dot, zr_dot)
        tr = tr_raw / (ca.norm_2(tr_raw) + 1e-8)

        # cost
        Q_c = ca.MX.sym("Q_c")
        Q_l = ca.MX.sym("Q_l")
        Q_s = ca.MX.sym("Q_s")
        Q_a = ca.MX.sym("Q_a")
        Q_sdd = ca.MX.sym("Q_sdd")

        e_tot = pos - pr
        e_l = ca.dot(tr, e_tot)
        e_l_vec = e_l * tr
        e_c = e_tot - e_l_vec

        cost_expr = (
            Q_c * ca.dot(e_c, e_c) + Q_l * e_l**2 - Q_s * s_dot + Q_sdd * s_ddot**2
        )

        cost_expr_e = Q_c * ca.dot(e_c, e_c) + Q_l * e_l**2 - Q_s * s_dot

        # clf
        f = ca.vertcat(vx, vy, vz, 0, 0, 0, s_dot, 0)
        g = ca.vertcat(
            ca.horzcat(0, 0, 0, 0),
            ca.horzcat(0, 0, 0, 0),
            ca.horzcat(0, 0, 0, 0),
            ca.horzcat(1, 0, 0, 0),
            ca.horzcat(0, 1, 0, 0),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 0),
            ca.horzcat(0, 0, 0, 1),
        )

        e_cdot = ca.jacobian(e_c, x) @ f
        e_ldot = ca.jacobian(e_l, x) @ f
        sc = e_cdot + 1 * e_c
        sl = e_ldot + 1 * e_l
        # e_c_norm = ca.norm_2(e_c)
        # e_l_norm = ca.norm_2(e_l)
        #
        # e_cdot = ca.jacobian(e_c_norm, x) @ f
        # e_ldot = ca.jacobian(e_l_norm, x) @ f
        #
        # sc = e_cdot + 1 * e_c_norm
        # sl = e_ldot + 1 * e_l_norm
        # lyap_f = 1.0 * sc**2 + 1.0 * sl**2
        lyap_f = 1.0 * sc.T @ sc + 1.0 * sl.T @ sl

        lfv = ca.jacobian(lyap_f, x) @ f
        lgv = ca.jacobian(lyap_f, x) @ g
        lgvu = lgv @ u
        v_dot = lfv + lgvu
        lyap_con = v_dot + 1.0 * lyap_f

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
            # e_axis_a_coeff,
            # e_axis_b_coeff,
            # e_offset_a_coeff,
            # e_offset_b_coeff,
            L_path,
            Q_c,
            Q_l,
            Q_s,
            Q_a,
            Q_sdd,
        )

        s_cons = s - L_path

        model = AcadosModel()
        # model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p

        # constraints
        model.con_h_expr_0 = ca.vertcat(s_cons, lyap_con)
        model.con_h_expr = ca.vertcat(s_cons, lyap_con)
        model.con_h_expr_e = ca.vertcat(s_cons)

        model.cost_expr_ext_cost = cost_expr
        model.cost_expr_ext_cost_e = cost_expr_e
        model.name = model_name

        # store meta information
        model.x_labels = [
            "$px$ [m]",
            "$py$ [m]",
            "$pz$ [m]",
            "$vx$ [m/s]",
            "$vy$ [m/s]",
            "$vz$ [m/s]",
            "$s$ []",
            "$sdot$ []",
        ]
        model.u_labels = ["$ax$", "$ay$", "$az$", "$sddot$"]
        model.t_label = "$t$ [s]"

        return model
