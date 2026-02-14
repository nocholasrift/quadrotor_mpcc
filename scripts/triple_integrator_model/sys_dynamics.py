import numpy as np
import casadi as ca
from acados_template import AcadosModel

class SysDyn():

    def __init__(self):
        pass

    def setup(self):

        model_name = "triple_integrator"

        # dynamics
        px = ca.MX.sym('px')
        py = ca.MX.sym('py')
        pz = ca.MX.sym('pz')
        p = ca.vertcat(px, py, pz)

        vx = ca.MX.sym('vx')
        vy = ca.MX.sym('vy')
        vz = ca.MX.sym('vz')
        v = ca.vertcat(vx, vy, vz)

        ax = ca.MX.sym('ax')
        ay = ca.MX.sym('ay')
        az = ca.MX.sym('az')
        a = ca.vertcat(ax, ay, az)


        s = ca.MX.sym('s')
        s_dot = ca.MX.sym('s_dot')

        x = ca.vertcat(p, v, a, s, s_dot)

        jx = ca.MX.sym('jx')
        jy = ca.MX.sym('jy')
        jz = ca.MX.sym('jz')
        j = ca.vertcat(jx, jy, jz)
        
        s_ddot = ca.MX.sym('s_ddot')

        u = ca.vertcat(j, s_ddot)

        px_dot = ca.MX.sym('px_dot')
        py_dot = ca.MX.sym('py_dot')
        pz_dot = ca.MX.sym('pz_dot')
        p_dot = ca.vertcat(px_dot, py_dot, pz_dot)

        vx_dot = ca.MX.sym('vx_dot')
        vy_dot = ca.MX.sym('vy_dot')
        vz_dot = ca.MX.sym('vz_dot')
        v_dot = ca.vertcat(vx_dot, vy_dot, vz_dot)

        ax_dot = ca.MX.sym('ax_dot')
        ay_dot = ca.MX.sym('ay_dot')
        az_dot = ca.MX.sym('az_dot')
        a_dot = ca.vertcat(ax_dot, ay_dot, az_dot)

        s1_dot = ca.MX.sym('s1_dot')
        s_dot_dot = ca.MX.sym('s_dot_dot')

        xdot = ca.vertcat(p_dot, v_dot, a_dot, s1_dot, s_dot_dot)
        f_expl = ca.vertcat(
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            jx,
            jy,
            jz,
            s_dot,
            s_ddot
        )

        f_impl = xdot - f_expl


        # tracking error 

        L_path = ca.MX.sym("L_path", 1)

        n_knots = 100
        arc_len_knots = np.linspace(0, 1, n_knots)

        xspl = ca.MX.sym("xspl", 1, 1)
        yspl = ca.MX.sym("yspl", 1, 1)
        zspl = ca.MX.sym("zspl", 1, 1)

        x_coeff = ca.MX.sym("x_coeffs", n_knots)
        y_coeff = ca.MX.sym("y_coeffs", n_knots)
        z_coeff = ca.MX.sym("z_coeffs", n_knots)

        interp_x = ca.interpolant("interp_x", "bspline", [arc_len_knots.tolist()])
        interp_exp_x = interp_x(xspl, x_coeff)
        xr_func = ca.Function("xr", [xspl, x_coeff], [interp_exp_x])

        interp_y = ca.interpolant("interp_y", "bspline", [arc_len_knots.tolist()])
        interp_exp_y = interp_y(yspl, y_coeff)
        yr_func = ca.Function("yr", [yspl, y_coeff], [interp_exp_y])

        interp_z = ca.interpolant("interp_z", "bspline", [arc_len_knots.tolist()])
        interp_exp_z = interp_z(zspl, z_coeff)
        zr_func = ca.Function("zr", [zspl, z_coeff], [interp_exp_z])

        s_norm = s / L_path
        xr = xr_func(s_norm, x_coeff)
        yr = yr_func(s_norm, y_coeff)
        zr = zr_func(s_norm, z_coeff)
        pr = ca.vertcat(xr, yr, zr)

        xr_dot = ca.jacobian(xr, s)
        yr_dot = ca.jacobian(yr, s)
        zr_dot = ca.jacobian(zr, s)
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

        e_tot_func = ca.Function("e_tot", [x_coeff, y_coeff, z_coeff, L_path, x], [e_tot])
        ec_func = ca.Function("ec", [x_coeff, y_coeff, z_coeff, L_path, x], [e_c])

        cost_expr = (
            Q_c * ca.dot(e_c, e_c)
            + Q_l * e_l**2
            + Q_j * ca.dot(j, j)
            + Q_sdd * s_ddot**2
            - Q_s * s_dot
        )

        cost_expr_e = Q_c * ca.dot(e_c, e_c) + Q_l * e_l**2 - Q_s * s_dot

        p = ca.vertcat(
            x_coeff,
            y_coeff,
            z_coeff,
            Q_c,
            Q_l,
            Q_j,
            Q_sdd,
            Q_s,
            L_path,
        )

        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p
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

        return model, e_tot_func, ec_func
