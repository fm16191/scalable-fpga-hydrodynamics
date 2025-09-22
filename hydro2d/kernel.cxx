#include "kernel.hpp"

static inline void convert_to_primitives(const size_t index, DATATYPE &U_rho, DATATYPE &inv_Q_rho, DATATYPE &U_ux,
                                         DATATYPE &U_uy, DATATYPE &U_p, DATATYPE &Q_rho, DATATYPE &Q_ux,
                                         DATATYPE &Q_uy, DATATYPE &Q_p, const DATATYPE *__restrict__ cache_U_rho,
                                         const DATATYPE *__restrict__ cache_U_u,
                                         const DATATYPE *__restrict__ cache_U_v,
                                         const DATATYPE *__restrict__ cache_U_E, const DATATYPE gamma_minus_one)
{
    // Conservative variables
    U_rho = cache_U_rho[index];
    U_ux  = cache_U_u[index];
    U_uy  = cache_U_v[index];
    U_p   = cache_U_E[index];

    // Primitive variables
    Q_rho = U_rho;
    inv_Q_rho = ONE / Q_rho;
    Q_ux = U_ux * inv_Q_rho;
    Q_uy = U_uy * inv_Q_rho;

    // e_int = U_p * inv_Q_rho - HALF * (Q_ux * Q_ux) - HALF * (Q_uy * Q_uy);
    /* */ [[intel::fpga_register]] DATATYPE temp_e_int = Q_ux*Q_ux + Q_uy*Q_uy;
    /* */ [[intel::fpga_register]] DATATYPE e_int = U_p * inv_Q_rho - HALF * temp_e_int;
    Q_p = gamma_minus_one * U_rho * e_int;
}

/*** Finite Volume Method (FVM) kernel for hydrodynamics kernel
 * Computes flux between the left and right cells. It can be in both X and Y directions,
 * left cell is either i-1 / j-1 or i / j, while right cell is either i / j or i+1 / j+1
 * @param Q_rho_left Primitive density value of the left cell
 * @param Q_ux_left Primitive x-axis velocity value of the left cell
 * @param Q_uy_left Primitive y-axis velocity value of the left cell
 * @param Q_p_left Primitive pressure value of the left cell
 * @param Q_rho_right Primitive density value of the right cell
 * @param Q_ux_right Primitive x-axis velocity value of the right cell
 * @param Q_uy_right Primitive y-axis velocity value of the right cell
 * @param Q_p_right Primitive pressure value of the right cell
 * @param K Safety coefficient
 * @param gamma Heat capacity ratio
 * @param divgamma 1 / (gamma - 1) coefficient
 * @param F_for_rho Return the computed flux for the density
 * @param F_for_ux Return the computed flux for the x-axis velocity
 * @param F_for_uy Return the computed flux for the y-axis velocity
 * @param F_for_E Return the computed flux for the pressure
 */
static void compute_fluxes(const DATATYPE Q_rho_left, const DATATYPE inv_Q_rho_left, const DATATYPE Q_ux_left,
                           const DATATYPE Q_uy_left, const DATATYPE Q_p_left, const DATATYPE Q_rho_right,
                           const DATATYPE inv_Q_rho_right, const DATATYPE Q_ux_right, const DATATYPE Q_uy_right,
                           const DATATYPE Q_p_right, const DATATYPE K, const DATATYPE gamma, const DATATYPE divgamma,
                           DATATYPE &F_for_rho, DATATYPE &F_for_ux, DATATYPE &F_for_uy, DATATYPE &F_for_E)
{
    DATATYPE Q_upwind_rho, Q_upwind_ux, Q_upwind_uy, Q_upwind_p;
    DATATYPE U_upwind_rho, U_upwind_ux, U_upwind_uy, U_upwind_p;

    // could pass here inv_Q_rho_left as already computed
    DATATYPE CSI_left = SQRT(gamma * Q_p_left * inv_Q_rho_left);
    DATATYPE a_left = Q_rho_left * CSI_left;

    // could pass here inv_Q_rho_right as already computed
    DATATYPE CSI_right = SQRT(gamma * Q_p_right * inv_Q_rho_right);
    DATATYPE a_right = Q_rho_right * CSI_right;

    DATATYPE a = K * MMAX(a_left, a_right);

    const DATATYPE diva = ONE / a;
    DATATYPE Q_u_star = HALF * ((Q_ux_left + Q_ux_right) - diva * (Q_p_right - Q_p_left));
    DATATYPE Q_p_star = HALF * ((Q_p_left + Q_p_right) - /* theta - assumed to be 1 */ a * (Q_ux_right - Q_ux_left));

    const bool up_ustar = Q_u_star >= ZERO;
Q_upwind_rho = up_ustar ? Q_rho_left : Q_rho_right;
    Q_upwind_ux  = up_ustar ? Q_ux_left : Q_ux_right;
    Q_upwind_uy  = up_ustar ? Q_uy_left : Q_uy_right;
    Q_upwind_p   = up_ustar ? Q_p_left : Q_p_right;

    // Compute back the conservative variables
    U_upwind_rho = Q_upwind_rho;
    U_upwind_ux  = Q_upwind_ux * Q_upwind_rho;
    U_upwind_uy  = Q_upwind_uy * Q_upwind_rho;
    U_upwind_p   = Q_upwind_p * divgamma + HALF * Q_upwind_rho * (Q_upwind_ux*Q_upwind_ux + Q_upwind_uy*Q_upwind_uy);

    // Compute fluxes
    F_for_rho = U_upwind_rho * Q_u_star /*  */ + ZERO;
    F_for_ux  = U_upwind_ux  * Q_u_star /*  */ + Q_p_star;
    F_for_uy  = U_upwind_uy  * Q_u_star /*  */ + ZERO;
    F_for_E   = U_upwind_p   * Q_u_star /*  */ + Q_p_star * Q_u_star;
}

/*** Finite Volume Method (FVM) kernel for hydrodynamics kernel
 * @param d_rho Current state of the density
 * @param d_u Current state of the x velocities
 * @param d_v Current state of the y velocities
 * @param d_E Current state of the pressure
 * @param d_rho_next Next state of the density (to be computed)
 * @param d_u_next Next state of the x velocities (to be computed)
 * @param d_v_next Next state of the y velocities (to be computed)
 * @param d_E_next Next state of the pressure (to be computed)
 * @param NB_X Block size of the x axis
 * @param NB_Y Block size of the y axis
 */
// static void kernel_hydro_fvm(const bool x_border_type, const DATATYPE *__restrict__ d_rho,
static void kernel_hydro_fvm(const DATATYPE *__restrict__ d_rhoE, const DATATYPE *__restrict__ d_uv,
                             DATATYPE *__restrict__ d_rhoE_next, DATATYPE *__restrict__ d_uv_next,
                             const DATATYPE K, const DATATYPE gamma, const DATATYPE gamma_minus_one,
                             const DATATYPE divgamma, const size_t &NB_X, const size_t &NB_Y, const DATATYPE &DtDx,
                             const DATATYPE &DtDy, const DATATYPE &min_spacing, const DATATYPE C,
                             DATATYPE *__restrict__ Dt_next)
{
    const size_t x_stride = NB_X + 2;
    const size_t y_stride = NB_Y + 2;
    const size_t max = x_stride * y_stride;

    DATATYPE cache_U_rho[CACHE_SIZE];
    DATATYPE cache_U_ux[CACHE_SIZE];
    DATATYPE cache_U_uy[CACHE_SIZE];
    DATATYPE cache_U_E[CACHE_SIZE];

    DATATYPE next_min_Dt = LITERAL(1e20);

    // Need to visit all cells except the opposite, in order to get ghost cells in cache.
    // Skip first and last cell, since this is a first order stencil.
    for (size_t k = 1; k < max - 1; ++k) {
        // size_t index_ipoj = k;
        size_t index_ij = k - y_stride;
        size_t cache_index_ipoj = k % CACHE_SIZE;
        cache_U_rho[cache_index_ipoj] = d_rhoE[2*k];
        cache_U_ux[cache_index_ipoj]  = d_uv[2*k];
        cache_U_uy[cache_index_ipoj]  = d_uv[2*k+1];
        cache_U_E[cache_index_ipoj]   = d_rhoE[2*k+1];

        // ipoj row and col
        const size_t row_pos = k / y_stride;
        const size_t col_pos = k % y_stride;

        // ROW-Major from top left.
        if (row_pos < 2) continue; // left border i < 2 border
        if (col_pos == 0) continue;            // top border ghost cells j == 0
        if (col_pos == y_stride - 1) continue; // bottom border ghost cells j = nb_y + 1

        [[intel::fpga_register]] DATATYPE fx1_rho = ZERO, fx1_ux = ZERO, fx1_uy = ZERO, fx1_E = ZERO;
        [[intel::fpga_register]] DATATYPE fy1_rho = ZERO, fy1_ux = ZERO, fy1_uy = ZERO, fy1_E = ZERO;

        [[intel::fpga_register]] DATATYPE fx2_rho = ZERO, fx2_ux = ZERO, fx2_uy = ZERO, fx2_E = ZERO;
        [[intel::fpga_register]] DATATYPE fy2_rho = ZERO, fy2_ux = ZERO, fy2_uy = ZERO, fy2_E = ZERO;

        [[intel::fpga_register]] DATATYPE U_rho_right, U_ux_right, U_uy_right, U_p_right;
        [[intel::fpga_register]] DATATYPE Q_rho_right, Q_ux_right, Q_uy_right, Q_p_right;
        [[intel::fpga_register]] DATATYPE inv_Q_rho_right;

        // x_border_type = 0 means left sub_domain => disable right reflective boundary condition
        // x_border_type = 1 means right sub_domain => disable left reflective boundary condition

        size_t index_right = index_ij % CACHE_SIZE;
        convert_to_primitives(index_right, U_rho_right, inv_Q_rho_right, U_ux_right, U_uy_right, U_p_right,
                              Q_rho_right, Q_ux_right, Q_uy_right, Q_p_right, cache_U_rho, cache_U_ux, cache_U_uy,
                              cache_U_E, gamma_minus_one);

        // Compute fluxes
        /* First flux in X axis (i-1) */
        {
            [[intel::fpga_register]] DATATYPE U_rho_left, U_ux_left, U_uy_left, U_p_left;
            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left;
            [[intel::fpga_register]] DATATYPE inv_Q_rho_left;

            // bool is_left_wall = ((row_pos == 2) && (x_border_type != 1)); // Reflective left wall disabled when right sub_domain
            // bool is_left_wall = (row_pos == 2);
            // size_t index_left = is_left_wall ? index_right : (k - 2 * y_stride) % CACHE_SIZE;
            size_t index_left = (k - 2 * y_stride) % CACHE_SIZE;

            convert_to_primitives(index_left, U_rho_left, inv_Q_rho_left, U_ux_left, U_uy_left, U_p_left,
                                  Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left, cache_U_rho, cache_U_ux, cache_U_uy,
                                  cache_U_E, gamma_minus_one);
            // Q_ux_left = is_left_wall ? -Q_ux_left : Q_ux_left;

            compute_fluxes(Q_rho_left, inv_Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left, Q_rho_right,
                           inv_Q_rho_right, Q_ux_right, Q_uy_right, Q_p_right, K, gamma, divgamma, fx1_rho, fx1_ux,
                           fx1_uy, fx1_E);
        }

        /* Second flux in X axis (i+1) */
        {
            [[intel::fpga_register]] DATATYPE U_rho_left, U_ux_left, U_uy_left, U_p_left;
            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left;
            [[intel::fpga_register]] DATATYPE inv_Q_rho_left;

            // bool is_right_wall = ((row_pos == x_stride-1) && (x_border_type != 0)); // Reflective right wall disabled when left sub_domain
            // bool is_right_wall = (row_pos == x_stride-1);
            // size_t index_left = is_right_wall ? index_right : k % CACHE_SIZE;
            size_t index_left = k % CACHE_SIZE;

            convert_to_primitives(index_left, U_rho_left, inv_Q_rho_left, U_ux_left, U_uy_left, U_p_left,
                                  Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left, cache_U_rho, cache_U_ux, cache_U_uy,
                                  cache_U_E, gamma_minus_one);
            // Q_ux_left = is_right_wall ? -Q_ux_left : Q_ux_left;

            // Care, invert left & right, since right is our ij cell.
            compute_fluxes(Q_rho_right, inv_Q_rho_right, Q_ux_right, Q_uy_right, Q_p_right, Q_rho_left,
                           inv_Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left, K, gamma, divgamma, fx2_rho, fx2_ux,
                           fx2_uy, fx2_E);
        }

        /* First flux in Y axis (j-1) */
        {
            [[intel::fpga_register]] DATATYPE U_rho_left, U_ux_left, U_uy_left, U_p_left;
            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left;
            [[intel::fpga_register]] DATATYPE inv_Q_rho_left;

            bool is_top_wall = (col_pos == 1);
            size_t index_left = is_top_wall ? index_right : (k - y_stride - 1) % CACHE_SIZE;

            convert_to_primitives(index_left, U_rho_left, inv_Q_rho_left, U_ux_left, U_uy_left, U_p_left,
                                  Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left, cache_U_rho, cache_U_ux, cache_U_uy,
                                  cache_U_E, gamma_minus_one);

            Q_uy_left = is_top_wall ? -Q_uy_left : Q_uy_left;

            // Care, switching UX and UY
            compute_fluxes(Q_rho_left, inv_Q_rho_left, Q_uy_left, Q_ux_left, Q_p_left, Q_rho_right,
                           inv_Q_rho_right, Q_uy_right, Q_ux_right, Q_p_right, K, gamma, divgamma, fy1_rho, fy1_ux,
                           fy1_uy, fy1_E);
        }

        /* Second flux in Y axis (j+1) */
        {
            [[intel::fpga_register]] DATATYPE U_rho_left, U_ux_left, U_uy_left, U_p_left;
            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left;
            [[intel::fpga_register]] DATATYPE inv_Q_rho_left;

            bool is_bottom_wall = (col_pos == y_stride-2);
            size_t index_left = is_bottom_wall ? index_right : (k - y_stride + 1) % CACHE_SIZE;
            convert_to_primitives(index_left, U_rho_left, inv_Q_rho_left, U_ux_left, U_uy_left, U_p_left,
                                  Q_rho_left, Q_ux_left, Q_uy_left, Q_p_left, cache_U_rho, cache_U_ux, cache_U_uy,
                                  cache_U_E, gamma_minus_one);

            Q_uy_left = is_bottom_wall ? -Q_uy_left : Q_uy_left;

            // Care, switching UX and UY
            // Care, invert left & right, since right is our ij cell.
            compute_fluxes(Q_rho_right, inv_Q_rho_right, Q_uy_right, Q_ux_right, Q_p_right, Q_rho_left,
                           inv_Q_rho_left, Q_uy_left, Q_ux_left, Q_p_left, K, gamma, divgamma, fy2_rho, fy2_ux,
                           fy2_uy, fy2_E);
        }

        // Here we have finished updating the cell -> therefore we can do our store operation. All others cells
        // *should* be in cache for good access latency. index = (i - 1) * stride + j;
        size_t index_next = index_ij;
        // Care, we also inverted left & right for j+1 and i+1 fluxes, so need to minus in fx2 and fy2.

        [[intel::fpga_register]] DATATYPE U_rho_next, U_u_next, U_v_next, U_E_next;

        DATATYPE x_rho = DtDx * (fx1_rho - fx2_rho);
        DATATYPE x_ux  = DtDx * (fx1_ux  - fx2_ux );
        DATATYPE x_uy  = DtDx * (fx1_uy  - fx2_uy );
        DATATYPE x_E   = DtDx * (fx1_E   - fx2_E  );

        DATATYPE y_rho = DtDy * (fy1_rho - fy2_rho);
        DATATYPE y_ux  = DtDy * (fy1_uy  - fy2_uy );
        DATATYPE y_uy  = DtDy * (fy1_ux  - fy2_ux );
        DATATYPE y_E   = DtDy * (fy1_E   - fy2_E  );

        U_rho_next = U_rho_right + x_rho + y_rho;
        U_u_next   = U_ux_right  + x_ux  + y_ux ;
        U_v_next   = U_uy_right  + x_uy  + y_uy ;
        U_E_next   = U_p_right   + x_E   + y_E  ;

        d_rhoE_next[2*index_next] = U_rho_next;
        d_uv_next[2*index_next]   = U_u_next;
        d_uv_next[2*index_next+1]   = U_v_next;
        d_rhoE_next[2*index_next+1]   = U_E_next;

        // now compute Dt_next
        {
            [[intel::fpga_register]] DATATYPE U_rho, U_ux, U_uy, U_p;
            [[intel::fpga_register]] DATATYPE Q_rho, Q_ux, Q_uy, Q_p;
            [[intel::fpga_register]] DATATYPE inv_Q_rho;

            // Conservative variables
            U_rho = U_rho_next;
            U_ux  = U_u_next;
            U_uy  = U_v_next;
            U_p   = U_E_next;

            // Primitive variables
            Q_rho = U_rho;
            inv_Q_rho = ONE / Q_rho;
            Q_ux = U_ux * inv_Q_rho;
            Q_uy = U_uy * inv_Q_rho;

            // e_int = U_p * inv_Q_rho - HALF * (Q_ux * Q_ux) - HALF * (Q_uy * Q_uy);
            [[intel::fpga_register]] DATATYPE temp_e_int = Q_ux*Q_ux + Q_uy*Q_uy;
            [[intel::fpga_register]] DATATYPE e_int = U_p * inv_Q_rho - HALF * temp_e_int;
            Q_p = gamma_minus_one * Q_rho * e_int;

            // compute speed of sound
            DATATYPE c_s = SQRT(gamma * Q_p * inv_Q_rho);

            // Compute Dt
            DATATYPE Unorm = SQRT(temp_e_int);
            DATATYPE next_local_Dt = min_spacing / (c_s + Unorm);

            if (next_local_Dt < next_min_Dt) next_min_Dt = next_local_Dt;
        }
    }

    *Dt_next = C * next_min_Dt;
}

/*** Launcher for the device kernels (update boundary conditions, Dt computation, Hydro computation)
 * @param d_rho Current state of the density
 * @param d_u Current state of the x velocities
 * @param d_v Current state of the y velocities
 * @param d_E Current state of the pressure
 * @param NB_X Block size of the x axis
 * @param NB_Y Block size of the y axis
 * @param queue the device queue
 */
// extern "C" double launcher(const bool x_border_type, DATATYPE *__restrict__ d_rho, DATATYPE *__restrict__ d_u, DATATYPE *__restrict__ d_v,
extern "C" double launcher(DATATYPE *__restrict__ d_rhoE, DATATYPE *__restrict__ d_uv,
                           DATATYPE *__restrict__ d_rhoE_next,
                           DATATYPE *__restrict__ d_uv_next, DATATYPE *__restrict__ Dt_next, const DATATYPE C,
                           const DATATYPE gamma, const DATATYPE gamma_minus_one, const DATATYPE divgamma,
                           const DATATYPE K, const size_t NB_X, const size_t NB_Y, const DATATYPE &DtDx,
                           const DATATYPE &DtDy, const DATATYPE &min_spacing, sycl::queue queue)
{
    struct timespec fpga_hydro_compute_t1, fpga_hydro_compute_t2;

#ifdef REPORT
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // kernel_hydro_fvm(x_border_type, d_rho, d_u, d_v, d_E, d_rho_next, d_u_next, d_v_next, d_E_next, K, gamma,
            kernel_hydro_fvm(d_rhoE, d_uv, d_rhoE_next, d_uv_next, K, gamma,
                             gamma_minus_one, divgamma, NB_X, NB_Y, DtDx, DtDy, min_spacing, C, Dt_next);
        });
    });
    queue.wait();
#else

    /* --------------------- */
    /* - Hydro computation - */
    /* --------------------- */
    clock_gettime(CLOCK_MONOTONIC, &fpga_hydro_compute_t1);
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // kernel_hydro_fvm(x_border_type, d_rho, d_u, d_v, d_E, d_rho_next, d_u_next, d_v_next, d_E_next, K, gamma,
            kernel_hydro_fvm(d_rhoE, d_uv, d_rhoE_next, d_uv_next, K, gamma,
                             gamma_minus_one, divgamma, NB_X, NB_Y, DtDx, DtDy, min_spacing, C, Dt_next);
        });
    });
    queue.wait();
    clock_gettime(CLOCK_MONOTONIC, &fpga_hydro_compute_t2);
#endif

    // Timers
    double fpga_hydro_compute = get_time_us(fpga_hydro_compute_t1, fpga_hydro_compute_t2);
    return fpga_hydro_compute;
}
