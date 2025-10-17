#include "kernel.hpp"

constexpr size_t ghost_size = 1;

static inline void convert_to_primitives_get_U(const size_t index, DATATYPE &U_rho, DATATYPE &U_ux, DATATYPE &U_uy,
                                         DATATYPE &U_uz, DATATYPE &U_p, DATATYPE &Q_rho, DATATYPE &Q_ux,
                                         DATATYPE &Q_uy, DATATYPE &Q_uz, DATATYPE &Q_p,
                                         const DATATYPE *__restrict__ cache_U_rho,
                                         const DATATYPE *__restrict__ cache_U_u,
                                         const DATATYPE *__restrict__ cache_U_v,
                                         const DATATYPE *__restrict__ cache_U_w,
                                         const DATATYPE *__restrict__ cache_U_E, const DATATYPE gamma_minus_one)
{
    // Conservative variables
    U_rho = cache_U_rho[index];
    U_ux  = cache_U_u[index];
    U_uy  = cache_U_v[index];
    U_uz  = cache_U_w[index];
    U_p   = cache_U_E[index];

    // Primitive variables
    Q_rho = U_rho;
    [[intel::fpga_register]] DATATYPE inv_Q_rho = ONE / Q_rho;
    Q_ux = U_ux * inv_Q_rho;
    Q_uy = U_uy * inv_Q_rho;
    Q_uz = U_uz * inv_Q_rho;

    // e_int = U_p * inv_Q_rho - HALF * (Q_ux * Q_ux) - HALF * (Q_uy * Q_uy);
    [[intel::fpga_register]] DATATYPE temp_e_int = Q_ux*Q_ux + Q_uy*Q_uy + Q_uz*Q_uz;
    [[intel::fpga_register]] DATATYPE e_int = U_p * inv_Q_rho - HALF * temp_e_int;
    Q_p = gamma_minus_one * U_rho * e_int;
}

static inline void convert_to_primitives(const size_t index, DATATYPE &Q_rho, DATATYPE &Q_ux, DATATYPE &Q_uy,
                                         DATATYPE &Q_uz, DATATYPE &Q_p, const DATATYPE *__restrict__ cache_U_rho,
                                         const DATATYPE *__restrict__ cache_U_u,
                                         const DATATYPE *__restrict__ cache_U_v,
                                         const DATATYPE *__restrict__ cache_U_w,
                                         const DATATYPE *__restrict__ cache_U_E, const DATATYPE gamma_minus_one)
{
    // Conservative variables
    [[intel::fpga_register]] DATATYPE U_rho = cache_U_rho[index];
    [[intel::fpga_register]] DATATYPE U_ux  = cache_U_u[index];
    [[intel::fpga_register]] DATATYPE U_uy  = cache_U_v[index];
    [[intel::fpga_register]] DATATYPE U_uz  = cache_U_w[index];
    [[intel::fpga_register]] DATATYPE U_p   = cache_U_E[index];

    // Primitive variables
    Q_rho = U_rho;
    [[intel::fpga_register]] DATATYPE inv_Q_rho = ONE / Q_rho;
    Q_ux = U_ux * inv_Q_rho;
    Q_uy = U_uy * inv_Q_rho;
    Q_uz = U_uz * inv_Q_rho;

    // e_int = U_p * inv_Q_rho - HALF * (Q_ux * Q_ux) - HALF * (Q_uy * Q_uy);
    [[intel::fpga_register]] DATATYPE temp_e_int = Q_ux*Q_ux + Q_uy*Q_uy + Q_uz*Q_uz;
    [[intel::fpga_register]] DATATYPE e_int = U_p * inv_Q_rho - HALF * temp_e_int;
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
static void compute_fluxes(const DATATYPE Q_rho_left, const DATATYPE Q_ux_left, const DATATYPE Q_uy_left,
                           const DATATYPE Q_uz_left, const DATATYPE Q_p_left, const DATATYPE Q_rho_right,
                           const DATATYPE Q_ux_right, const DATATYPE Q_uy_right, const DATATYPE Q_uz_right,
                           const DATATYPE Q_p_right, const DATATYPE K, const DATATYPE gamma,
                           const DATATYPE divgamma, DATATYPE &F_for_rho, DATATYPE &F_for_ux, DATATYPE &F_for_uy,
                           DATATYPE &F_for_uz, DATATYPE &F_for_E)
{
    [[intel::fpga_register]] DATATYPE Q_upwind_rho, Q_upwind_ux, Q_upwind_uy, Q_upwind_uz, Q_upwind_p;
    [[intel::fpga_register]] DATATYPE U_upwind_rho, U_upwind_ux, U_upwind_uy, U_upwind_uz, U_upwind_p;

    // a_left = Q_rho_left * sqrt(gamma * Q_p_left / Q_rho_left)
    //        = sqrt(gamma * Q_p_left * Q_rho_left)
    // if Q_rho is always >= 0
    // it never had been 0 otherwise the division would have failed, and density shouldn't be negative.

    //        = b * sqrt(a / b)
    //        = sqrt(a * b)
    // if b >= 0 always

    DATATYPE a_left = SQRT(gamma * Q_p_left * Q_rho_left);
    DATATYPE a_right = SQRT(gamma * Q_p_right * Q_rho_right);

    DATATYPE a = K * MMAX(a_left, a_right);

    const DATATYPE diva = ONE / a;
    DATATYPE Q_u_star = HALF * ((Q_ux_left + Q_ux_right) - diva * (Q_p_right - Q_p_left));
    DATATYPE Q_p_star = HALF * ((Q_p_left + Q_p_right) - /* theta - assumed to be 1 */ a * (Q_ux_right - Q_ux_left));

    const bool up_ustar = Q_u_star >= ZERO; 
    Q_upwind_rho = up_ustar ? Q_rho_left : Q_rho_right;
    Q_upwind_ux  = up_ustar ? Q_ux_left : Q_ux_right;
    Q_upwind_uy  = up_ustar ? Q_uy_left : Q_uy_right;
    Q_upwind_uz  = up_ustar ? Q_uz_left : Q_uz_right;
    Q_upwind_p   = up_ustar ? Q_p_left : Q_p_right;

    // Compute back the conservative variables
    U_upwind_rho = Q_upwind_rho;
    U_upwind_ux  = Q_upwind_ux * Q_upwind_rho;
    U_upwind_uy  = Q_upwind_uy * Q_upwind_rho;
    U_upwind_uz  = Q_upwind_uz * Q_upwind_rho;

    DATATYPE eint = Q_upwind_p * divgamma;
    DATATYPE ekin = HALF * Q_upwind_rho * (Q_upwind_ux*Q_upwind_ux + Q_upwind_uy*Q_upwind_uy + Q_upwind_uz*Q_upwind_uz);
    U_upwind_p    = eint + ekin;

    // Compute fluxes
    F_for_rho = U_upwind_rho * Q_u_star /*  */ + ZERO;
    F_for_ux  = U_upwind_ux  * Q_u_star /*  */ + Q_p_star;
    F_for_uy  = U_upwind_uy  * Q_u_star /*  */ + ZERO;
    F_for_uz  = U_upwind_uz  * Q_u_star /*  */ + ZERO;
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
 * @param d_w_next Next state of the z velocities (to be computed)
 * @param d_E_next Next state of the pressure (to be computed)
 * @param NB_X Block size of the x axis
 * @param NB_Y Block size of the y axis
 */
static void kernel_hydro3d_fvm(const DATATYPE *__restrict__ d_rhoE, const DATATYPE *__restrict__ d_uvw,
                               DATATYPE *__restrict__ d_rhoE_next, DATATYPE *__restrict__ d_uvw_next,
                               const DATATYPE K, const DATATYPE gamma, const DATATYPE gamma_minus_one,
                               const DATATYPE divgamma, const size_t &NB_X, const size_t &NB_Y, const size_t &NB_Z,
                               const DATATYPE &DtDx, const DATATYPE &DtDy, const DATATYPE &DtDz,
                               const DATATYPE &min_spacing, const DATATYPE C, DATATYPE *__restrict__ Dt_next)
{
    const size_t x_stride = NB_X + 2;
    const size_t y_stride = NB_Y + 2;
    const size_t z_stride = NB_Z + 2;
    const size_t zy_stride = y_stride * z_stride;
    const size_t max = x_stride * y_stride * z_stride;

    DATATYPE cache_U_rho[CACHE_SIZE];
    DATATYPE cache_U_ux[CACHE_SIZE];
    DATATYPE cache_U_uy[CACHE_SIZE];
    DATATYPE cache_U_uz[CACHE_SIZE];
    DATATYPE cache_U_E[CACHE_SIZE];

    DATATYPE next_min_Dt = LITERAL(1e20);

    constexpr unsigned MIN_ACC = 4; // power of two, tune to II target / latency
    [[intel::fpga_register]] double min_acc[MIN_ACC];
    for (unsigned t = 0; t < MIN_ACC; ++t) min_acc[t] = LITERAL(1e20);

    for (size_t l = 0; l < max; ++l){
        // size_t index_ipojk = (i+1) * zy_stride + j * z_stride + k;
        size_t index_ipojk = l;
        size_t index_ijk = l - zy_stride;
        size_t cache_index_ipojk = index_ipojk & (CACHE_SIZE - 1);
        cache_U_rho[cache_index_ipojk] = d_rhoE[2*index_ipojk];
        cache_U_ux[cache_index_ipojk]  = d_uvw[3*index_ipojk];
        cache_U_uy[cache_index_ipojk]  = d_uvw[3*index_ipojk+1];
        cache_U_uz[cache_index_ipojk]  = d_uvw[3*index_ipojk+2];
        cache_U_E[cache_index_ipojk]   = d_rhoE[2*index_ipojk+1];

        const size_t row_pos = l / zy_stride;
        const size_t col_pos = l % zy_stride / z_stride;
        const size_t dep_pos = l % z_stride;

        // ROW-Major from top left.
        if (row_pos < 2) continue;             // Left face : can't compute until i+1
        if (col_pos == 0) continue;            // Bottom face : ignore computation for ghost cells
        if (col_pos == y_stride - 1) continue; // Top face : ignore computation for ghost cells
        if (dep_pos == 0) continue;            // Front face : ignore computation for ghost cells
        if (dep_pos == z_stride - 1) continue; // Back face : ignore computation for ghost cells

        [[intel::fpga_register]] DATATYPE fx1_rho = ZERO, fx1_ux = ZERO, fx1_uy = ZERO, fx1_uz = ZERO, fx1_E = ZERO;
        [[intel::fpga_register]] DATATYPE fy1_rho = ZERO, fy1_ux = ZERO, fy1_uy = ZERO, fy1_uz = ZERO, fy1_E = ZERO;
        [[intel::fpga_register]] DATATYPE fz1_rho = ZERO, fz1_ux = ZERO, fz1_uy = ZERO, fz1_uz = ZERO, fz1_E = ZERO;

        [[intel::fpga_register]] DATATYPE fx2_rho = ZERO, fx2_ux = ZERO, fx2_uy = ZERO, fx2_uz = ZERO, fx2_E = ZERO;
        [[intel::fpga_register]] DATATYPE fy2_rho = ZERO, fy2_ux = ZERO, fy2_uy = ZERO, fy2_uz = ZERO, fy2_E = ZERO;
        [[intel::fpga_register]] DATATYPE fz2_rho = ZERO, fz2_ux = ZERO, fz2_uy = ZERO, fz2_uz = ZERO, fz2_E = ZERO;

        [[intel::fpga_register]] DATATYPE U_rho_right, U_ux_right, U_uy_right, U_uz_right, U_p_right;
        [[intel::fpga_register]] DATATYPE Q_rho_right, Q_ux_right, Q_uy_right, Q_uz_right, Q_p_right;

        size_t index_right = index_ijk & (CACHE_SIZE - 1);
        convert_to_primitives_get_U(index_right, U_rho_right, U_ux_right, U_uy_right, U_uz_right, U_p_right,
                                    Q_rho_right, Q_ux_right, Q_uy_right, Q_uz_right, Q_p_right, cache_U_rho,
                                    cache_U_ux, cache_U_uy, cache_U_uz, cache_U_E, gamma_minus_one);

        // Compute fluxes
        /* First flux in X axis (i-1) */
        {
            size_t index_left = (index_ipojk - 2 * zy_stride) & (CACHE_SIZE - 1);

            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left;

            convert_to_primitives(index_left, Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left, cache_U_rho,
                                  cache_U_ux, cache_U_uy, cache_U_uz, cache_U_E, gamma_minus_one);

            compute_fluxes(Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left, Q_rho_right, Q_ux_right,
                           Q_uy_right, Q_uz_right, Q_p_right, K, gamma, divgamma, fx1_rho, fx1_ux, fx1_uy, fx1_uz,
                           fx1_E);
        }

        /* Second flux in X axis (i+1) */
        {
            size_t index_left = index_ipojk & (CACHE_SIZE - 1);

            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left;

            convert_to_primitives(index_left, Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left, cache_U_rho,
                                  cache_U_ux, cache_U_uy, cache_U_uz, cache_U_E, gamma_minus_one);

            compute_fluxes(Q_rho_right, Q_ux_right, Q_uy_right, Q_uz_right, Q_p_right, Q_rho_left, Q_ux_left,
                           Q_uy_left, Q_uz_left, Q_p_left, K, gamma, divgamma, fx2_rho, fx2_ux, fx2_uy, fx2_uz,
                           fx2_E);
        }

        /* First flux in Y axis (j-1) */
        {
            bool is_top_wall = (col_pos == 1);
            size_t index_left = (index_ijk - z_stride) & (CACHE_SIZE - 1);

            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left;

            convert_to_primitives(index_left, Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left, cache_U_rho,
                                  cache_U_ux, cache_U_uy, cache_U_uz, cache_U_E, gamma_minus_one);

            [[intel::fpga_register]] DATATYPE Q_rho_left_used = is_top_wall ?  Q_rho_right : Q_rho_left;
            [[intel::fpga_register]] DATATYPE Q_uy_left_used  = is_top_wall ? -Q_uy_right  : Q_uy_left; // negate for wall
            [[intel::fpga_register]] DATATYPE Q_ux_left_used  = is_top_wall ?  Q_ux_right  : Q_ux_left;
            [[intel::fpga_register]] DATATYPE Q_uz_left_used  = is_top_wall ?  Q_uz_right  : Q_uz_left;
            [[intel::fpga_register]] DATATYPE Q_p_left_used   = is_top_wall ?  Q_p_right   : Q_p_left;

            compute_fluxes(Q_rho_left_used, Q_uy_left_used, Q_ux_left_used, Q_uz_left_used, Q_p_left_used,
                           Q_rho_right, Q_uy_right, Q_ux_right, Q_uz_right, Q_p_right, K, gamma, divgamma, fy1_rho,
                           fy1_ux, fy1_uy, fy1_uz, fy1_E);
        }

        /* Second flux in Y axis (j+1) */
        {
            bool is_bottom_wall = (col_pos == y_stride - ghost_size - 1);
            size_t index_left = (index_ijk + z_stride) & (CACHE_SIZE - 1);

            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left;

            convert_to_primitives(index_left, Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left, cache_U_rho,
                                  cache_U_ux, cache_U_uy, cache_U_uz, cache_U_E, gamma_minus_one);

            [[intel::fpga_register]] DATATYPE Q_rho_left_used = is_bottom_wall ?  Q_rho_right : Q_rho_left;
            [[intel::fpga_register]] DATATYPE Q_ux_left_used  = is_bottom_wall ?  Q_ux_right  : Q_ux_left;
            [[intel::fpga_register]] DATATYPE Q_uy_left_used  = is_bottom_wall ? -Q_uy_right  : Q_uy_left; // negate for wall
            [[intel::fpga_register]] DATATYPE Q_uz_left_used  = is_bottom_wall ?  Q_uz_right  : Q_uz_left;
            [[intel::fpga_register]] DATATYPE Q_p_left_used   = is_bottom_wall ?  Q_p_right   : Q_p_left;

            compute_fluxes(Q_rho_right, Q_uy_right, Q_ux_right, Q_uz_right, Q_p_right, Q_rho_left_used,
                           Q_uy_left_used, Q_ux_left_used, Q_uz_left_used, Q_p_left_used, K, gamma, divgamma,
                           fy2_rho, fy2_ux, fy2_uy, fy2_uz, fy2_E);
        }

        /* First flux in Z axis (k-1) */
        {
            bool is_back_wall = (dep_pos == 1);
            size_t index_left = (index_ijk - 1) & (CACHE_SIZE - 1);

            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left;

            convert_to_primitives(index_left, Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left, cache_U_rho,
                                  cache_U_ux, cache_U_uy, cache_U_uz, cache_U_E, gamma_minus_one);

            [[intel::fpga_register]] DATATYPE Q_rho_left_used = is_back_wall ?  Q_rho_right : Q_rho_left;
            [[intel::fpga_register]] DATATYPE Q_uz_left_used  = is_back_wall ? -Q_uz_right  : Q_uz_left; // negate for wall
            [[intel::fpga_register]] DATATYPE Q_ux_left_used  = is_back_wall ?  Q_ux_right  : Q_ux_left;
            [[intel::fpga_register]] DATATYPE Q_uy_left_used  = is_back_wall ?  Q_uy_right  : Q_uy_left;
            [[intel::fpga_register]] DATATYPE Q_p_left_used   = is_back_wall ?  Q_p_right   : Q_p_left;

            compute_fluxes(Q_rho_left_used, Q_uz_left_used, Q_uy_left_used, Q_ux_left_used, Q_p_left_used,
                           Q_rho_right, Q_uz_right, Q_uy_right, Q_ux_right, Q_p_right, K, gamma, divgamma, fz1_rho,
                           fz1_ux, fz1_uy, fz1_uz, fz1_E);
        }

        /* Second flux in Z axis (k+1) */
        {
            bool is_front_wall = (dep_pos == z_stride - ghost_size - 1);
            size_t index_left = (index_ijk + 1) & (CACHE_SIZE - 1);

            [[intel::fpga_register]] DATATYPE Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left;

            convert_to_primitives(index_left, Q_rho_left, Q_ux_left, Q_uy_left, Q_uz_left, Q_p_left, cache_U_rho,
                                  cache_U_ux, cache_U_uy, cache_U_uz, cache_U_E, gamma_minus_one);

            [[intel::fpga_register]] DATATYPE Q_rho_left_used = is_front_wall ?  Q_rho_right : Q_rho_left;
            [[intel::fpga_register]] DATATYPE Q_uz_left_used  = is_front_wall ? -Q_uz_right  : Q_uz_left; // negate for wall
            [[intel::fpga_register]] DATATYPE Q_ux_left_used  = is_front_wall ?  Q_ux_right  : Q_ux_left;
            [[intel::fpga_register]] DATATYPE Q_uy_left_used  = is_front_wall ?  Q_uy_right  : Q_uy_left;
            [[intel::fpga_register]] DATATYPE Q_p_left_used   = is_front_wall ?  Q_p_right   : Q_p_left;

            compute_fluxes(Q_rho_right, Q_uz_right, Q_uy_right, Q_ux_right, Q_p_right, Q_rho_left_used,
                           Q_uz_left_used, Q_uy_left_used, Q_ux_left_used, Q_p_left_used, K, gamma, divgamma,
                           fz2_rho, fz2_ux, fz2_uy, fz2_uz, fz2_E);
        }

        // Here we have finished updating the cell -> therefore we can do our store operation. All others cells
        // *should* be in cache for good access latency.
        const size_t index_next = index_ijk;
        // Care, we also inverted left & right for j+1 and i+1 fluxes, so need to minus fx2, fy2 and fz2 contributions.

        // Next conservative values
        [[intel::fpga_register]] DATATYPE U_rho_next, U_u_next, U_v_next, U_w_next, U_E_next;

        DATATYPE x_rho = DtDx * (fx1_rho - fx2_rho);
        DATATYPE x_ux  = DtDx * (fx1_ux  - fx2_ux );
        DATATYPE x_uy  = DtDx * (fx1_uy  - fx2_uy );
        DATATYPE x_uz  = DtDx * (fx1_uz  - fx2_uz );
        DATATYPE x_E   = DtDx * (fx1_E   - fx2_E  );

        DATATYPE y_rho = DtDy * (fy1_rho - fy2_rho );
        DATATYPE y_ux  = DtDy * (fy1_uy  - fy2_uy  );
        DATATYPE y_uy  = DtDy * (fy1_ux  - fy2_ux  );
        DATATYPE y_uz  = DtDy * (fy1_uz  - fy2_uz  );
        DATATYPE y_E   = DtDy * (fy1_E   - fy2_E   );

        DATATYPE z_rho = DtDz * (fz1_rho - fz2_rho);
        DATATYPE z_ux  = DtDz * (fz1_uz  - fz2_uz );
        DATATYPE z_uy  = DtDz * (fz1_uy  - fz2_uy );
        DATATYPE z_uz  = DtDz * (fz1_ux  - fz2_ux );
        DATATYPE z_E   = DtDz * (fz1_E   - fz2_E  );

        U_rho_next = U_rho_right + x_rho + y_rho + z_rho;
        U_u_next   = U_ux_right  + x_ux  + y_ux  + z_ux ;
        U_v_next   = U_uy_right  + x_uy  + y_uy  + z_uy ;
        U_w_next   = U_uz_right  + x_uz  + y_uz  + z_uz ;
        U_E_next   = U_p_right   + x_E   + y_E   + z_E  ;

        d_rhoE_next[2*index_next]   = U_rho_next;
        d_uvw_next[3*index_next]    = U_u_next;
        d_uvw_next[3*index_next+1]  = U_v_next;
        d_uvw_next[3*index_next+2]  = U_w_next;
        d_rhoE_next[2*index_next+1] = U_E_next;

        // now compute Dt_next
        {
            [[intel::fpga_register]] DATATYPE U_rho, U_ux, U_uy, U_uz, U_p;
            [[intel::fpga_register]] DATATYPE Q_rho, Q_ux, Q_uy, Q_uz, Q_p;
            [[intel::fpga_register]] DATATYPE inv_Q_rho;

            // Conservative variables
            U_rho = U_rho_next;
            U_ux  = U_u_next;
            U_uy  = U_v_next;
            U_uz  = U_w_next;
            U_p   = U_E_next;

            // Primitive variables
            Q_rho = U_rho;
            inv_Q_rho = ONE / Q_rho;
            Q_ux = U_ux * inv_Q_rho;
            Q_uy = U_uy * inv_Q_rho;
            Q_uz = U_uz * inv_Q_rho;

            [[intel::fpga_register]] DATATYPE temp_e_int = Q_ux*Q_ux + Q_uy*Q_uy + Q_uz*Q_uz;
            [[intel::fpga_register]] DATATYPE e_int = U_p * inv_Q_rho - HALF * temp_e_int;
            Q_p = gamma_minus_one * Q_rho * e_int;

            // compute speed of sound
            DATATYPE c_s = SQRT(gamma * Q_p * inv_Q_rho);

            // Compute Dt
            DATATYPE Unorm = SQRT(temp_e_int);
            DATATYPE next_local_Dt = min_spacing / (c_s + Unorm);

            if (next_local_Dt < next_min_Dt) next_min_Dt = next_local_Dt;

            // unsigned slot = static_cast<unsigned>(l & (MIN_ACC - 1));
            // double cur = min_acc[slot];
            // if (next_local_Dt < cur) min_acc[slot] = next_local_Dt;
        }
    }

    // next_min_Dt = min_acc[0];
    // for (unsigned t = 1; t < MIN_ACC; ++t)
    //     if (min_acc[t] < next_min_Dt) next_min_Dt = min_acc[t];

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
extern "C" double launcher(DATATYPE *__restrict__ d_rhoE, DATATYPE *__restrict__ d_uvw,
                           DATATYPE *__restrict__ d_rhoE_next, DATATYPE *__restrict__ d_uvw_next,
                           DATATYPE *__restrict__ Dt_next, const DATATYPE C, const DATATYPE gamma,
                           const DATATYPE gamma_minus_one, const DATATYPE divgamma, const DATATYPE K,
                           const size_t NB_X, const size_t NB_Y, const size_t NB_Z, const DATATYPE &DtDx,
                           const DATATYPE &DtDy, const DATATYPE &DtDz, const DATATYPE &min_spacing,
                           sycl::queue queue)
{
    struct timespec fpga_hydro3d_compute_t1, fpga_hydro3d_compute_t2;

    #ifdef REPORT
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            kernel_hydro3d_fvm(d_rhoE, d_uvw, d_rhoE_next, d_uvw_next, K, gamma, gamma_minus_one, divgamma, NB_X,
                               NB_Y, NB_Z, DtDx, DtDy, DtDz, min_spacing, C, Dt_next);
        });
    });
    queue.wait();
#else

    /* --------------------- */
    /* - Hydro computation - */
    /* --------------------- */
    clock_gettime(CLOCK_MONOTONIC, &fpga_hydro3d_compute_t1);
    queue.submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            kernel_hydro3d_fvm(d_rhoE, d_uvw, d_rhoE_next, d_uvw_next, K, gamma, gamma_minus_one, divgamma, NB_X,
                               NB_Y, NB_Z, DtDx, DtDy, DtDz, min_spacing, C, Dt_next);
        });
    });
    queue.wait();
    clock_gettime(CLOCK_MONOTONIC, &fpga_hydro3d_compute_t2);
#endif

    // Timers
    double fpga_hydro3d_compute = get_time_us(fpga_hydro3d_compute_t1, fpga_hydro3d_compute_t2);
    return fpga_hydro3d_compute;
}
