import numpy as np
import casadi as ca
from acados_template import AcadosModel
from common import n_knots


def create_interp(name, knots, coeffs):
    s = ca.MX.sym("s")
    interp = ca.interpolant(name, "bspline", [knots.tolist()])
    return ca.Function(name, [s, coeffs], [interp(s, coeffs)])


class SysDyn:

    def __init__(self):
        pass

    def setup(self):

        model_name = "triple_integrator"

        # dynamics
        px = ca.MX.sym("px")
        py = ca.MX.sym("py")
        pz = ca.MX.sym("pz")
        p = ca.vertcat(px, py, pz)

        vx = ca.MX.sym("vx")
        vy = ca.MX.sym("vy")
        vz = ca.MX.sym("vz")
        v = ca.vertcat(vx, vy, vz)

        ax = ca.MX.sym("ax")
        ay = ca.MX.sym("ay")
        az = ca.MX.sym("az")
        a = ca.vertcat(ax, ay, az)

        s = ca.MX.sym("s")
        s_dot = ca.MX.sym("s_dot")

        x = ca.vertcat(p, v, a, s, s_dot)

        jx = ca.MX.sym("jx")
        jy = ca.MX.sym("jy")
        jz = ca.MX.sym("jz")
        j = ca.vertcat(jx, jy, jz)

        s_ddot = ca.MX.sym("s_ddot")

        u = ca.vertcat(j, s_ddot)

        px_dot = ca.MX.sym("px_dot")
        py_dot = ca.MX.sym("py_dot")
        pz_dot = ca.MX.sym("pz_dot")
        p_dot = ca.vertcat(px_dot, py_dot, pz_dot)

        vx_dot = ca.MX.sym("vx_dot")
        vy_dot = ca.MX.sym("vy_dot")
        vz_dot = ca.MX.sym("vz_dot")
        v_dot = ca.vertcat(vx_dot, vy_dot, vz_dot)

        ax_dot = ca.MX.sym("ax_dot")
        ay_dot = ca.MX.sym("ay_dot")
        az_dot = ca.MX.sym("az_dot")
        a_dot = ca.vertcat(ax_dot, ay_dot, az_dot)

        s1_dot = ca.MX.sym("s1_dot")
        s_dot_dot = ca.MX.sym("s_dot_dot")

        xdot = ca.vertcat(p_dot, v_dot, a_dot, s1_dot, s_dot_dot)
        f_expl = ca.vertcat(vx, vy, vz, ax, ay, az, jx, jy, jz, s_dot, s_ddot)

        f_impl = xdot - f_expl

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
        tr = ca.vertcat(xr_dot, yr_dot, zr_dot)

        # cost
        Q_c = ca.MX.sym("Q_c")
        Q_l = ca.MX.sym("Q_l")
        Q_s = ca.MX.sym("Q_s")
        Q_j = ca.MX.sym("Q_j")
        Q_sdd = ca.MX.sym("Q_sdd")

        e_tot = p - pr
        e_l = ca.dot(tr, e_tot)
        e_l_vec = e_l * tr
        e_c = e_tot - e_l_vec

        cost_expr = (
            Q_c * ca.dot(e_c, e_c)
            + Q_l * e_l**2
            + Q_j * ca.dot(j, j)
            + Q_sdd * s_ddot**2
            - Q_s * s_dot
        )

        cost_expr_e = Q_c * ca.dot(e_c, e_c) + Q_l * e_l**2 - Q_s * s_dot

        # control barrier function
        e1x = e1x_func(s_norm, e1x_coeff)
        e1y = e1y_func(s_norm, e1y_coeff)
        e1z = e1z_func(s_norm, e1z_coeff)
        e1 = ca.vertcat(e1x, e1y, e1z)

        e2x = e2x_func(s_norm, e2x_coeff)
        e2y = e2y_func(s_norm, e2y_coeff)
        e2z = e2z_func(s_norm, e2z_coeff)
        e2 = ca.vertcat(e2x, e2y, e2z)

        q1 = ca.dot(e1, e_tot)
        q2 = ca.dot(e2, e_tot)

        F = ca.vertcat(vx, vy, vz, ax, ay, az, 0, 0, 0, s_dot, 0)

        G = ca.DM.zeros(11, 4)
        G[6, 0] = 1
        G[7, 1] = 1
        G[8, 2] = 1
        G[10, 3] = 1

        radius = 0.5  # m
        cbf = 1 - (q1 / radius) ** 2 - (q2 / radius) ** 2
        # grad_cbf = ca.jacobian(cbf, x)
        # Lfh = grad_cbf @ F
        # Lgh = grad_cbf @ G
        # cbf_cons = Lfh + Lgh @ u + 1 * cbf

        grad_h = ca.jacobian(cbf, x)
        Lf_h = grad_h @ F

        grad_Lf_h = ca.jacobian(Lf_h, x)
        Lf2_h = grad_Lf_h @ F

        grad_Lf2_h = ca.jacobian(Lf2_h, x)
        Lf3_h = grad_Lf2_h @ F
        LgLf2_h = grad_Lf2_h @ G  # <-- input appears here

        alpha0 = ca.MX.sym("alpha0")
        alpha1 = ca.MX.sym("alpha1")
        alpha2 = ca.MX.sym("alpha2")

        cbf_cons = (
            Lf3_h + LgLf2_h @ u + 3 * alpha2 * Lf2_h + 3 * alpha1 * Lf_h + alpha0 * cbf
        )

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
            Q_j,
            Q_sdd,
            Q_s,
            L_path,
            alpha0,
            alpha1,
            alpha2,
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
            ],
            [Lf3_h, LgLf2_h, Lf2_h, Lf_h, cbf],
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
            ],
            ["Lf3h", "LgLf2h", "Lf2h", "Lfh", "cbf"],
        )

        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p
        model.con_h_expr_0 = ca.vertcat(cbf_cons, s_cons)
        model.con_h_expr = ca.vertcat(cbf_cons, s_cons)
        model.con_h_expr_e = ca.vertcat(s_cons)
        model.cost_expr_ext_cost = cost_expr
        model.cost_expr_ext_cost_e = cost_expr_e
        model.xdot = xdot
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
