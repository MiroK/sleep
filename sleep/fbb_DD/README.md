# Fluid-Biot-Biot with Domain-Decomposition/Operator splitting approach

We solve

-div(sigma_f(u_f, p_f)) = 0    in Omega_f(t)
-div(u_f)               = 0

-div(sigma_p(eta_p, p_p))                   =  0    in Omega_p(t)
u_p + kappa*grad(p_p)                       =  0
d/d_t(s0*p_p + alpla*div(eta_p)) + div(u_p) =  0

with interface boundary conditions  on Gamma(t)

sigma_f.n_f + sigma_p.n_p = 0
-n_f.sigma_f.n_f = p_p
u_f.n_f + (d_dt(eta_p) + u_p).n_p = 0
u_f.tau_f + d_d(eta_p).tau_f = 0

The solution algorithm is based on solving fluid problem on Omega_f(t0),
transfering data to solid problem which is solved on Omega_p(t0). Then
eta_p together with driving conditions on the bottom fluid wall determine
the displacement in fluid domain eta_f. Using their respective displacements
both domains are updated to t0 + dt.

# Questions

- [ ] At the moment eta_f is vector and its bcs on bottom wall (boundary with
blood region) are also expected vectorial. The side boundaries grad(eta_f).n = 0.
Is this okay?

- [ ] Fluid problem is setup expecting pressure on left and pressure on right,
traction bc on the top and velocity bcs on the bottom. Okay?

- [ ] Solid problem has eta_p.n = 0 and u_p = 0 on left and right, for top
we set sigma_p.n_p = 0 and p_p = p_E and bottom has fixed displacement
and prescribed pressure. Okay?