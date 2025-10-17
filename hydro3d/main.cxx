#include "kernel.hpp"
#include "timers.h"

#include <chrono>
#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stddef.h>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <mpi.h>

using namespace sycl;

using std::cerr;
using std::cout;
using std::setprecision;
using std::sqrt;
using std::stoi;
using std::vector;
using std::chrono::high_resolution_clock;

using std::string;
using std::stringstream;

// Defaults
static size_t NB_SUBDOMAINS_X = 2; // Force 2 MPI Processes for now
static size_t NB_SUBDOMAINS_Y = 1;
static size_t NB_SUBDOMAINS_Z = 1;

static size_t SUBDOMAIN_SIZE_X;
static size_t SUBDOMAIN_SIZE_Y;
static size_t SUBDOMAIN_SIZE_Z;

static size_t NB_X = 200;
static size_t NB_Y = 200;
static size_t NB_Z = 100;
static double MAX_T = 1;

static size_t MAX_IT = 1000;
static size_t write_interval = 0;
static string out_filename = "output"; // default

static int rank, nprocs;
static bool single_device_mode = false;

constexpr size_t ghost_size = 1;

class sub_domain {
    public:
    sycl::queue &queue;

    size_t nb_x, nb_y, nb_z;             // Actual domain size (without ghost cells)
    size_t x_stride, y_stride, z_stride; // Domain dimensions including ghost cells
    size_t zy_stride;                    // Combined stride for z and y dimensions
    size_t domain_size;                  // Total elements including ghost cells (x_stride * y_stride * z_stride)
    size_t max_device_elements;          // Maximum device elements for alignment (size_max)
    size_t buffer_size_bytes;            // Size in bytes for data transfer (domain_size * sizeof(DATATYPE))
    size_t buffer_size_bytes2;           // Secondary buffer size (2 * domain_size * sizeof(DATATYPE))

    vector<DATATYPE> h_rhoE, h_uvw;

    // Device pointers - allocated with max_device_elements for
    // alignment/performance
    DATATYPE *d_rhoE = nullptr, *d_uvw = nullptr;
    DATATYPE *d_rhoE_next = nullptr, *d_uvw_next = nullptr;
    DATATYPE *Dt_next = nullptr;

    // Dummy constructor
    // Using std::optional would be a great improvement.
    sub_domain(sycl::queue &q) : queue(q), nb_x(0), nb_y(0), nb_z(0), x_stride(0), y_stride(0), z_stride(0),
    zy_stride(0), domain_size(0), max_device_elements(0), buffer_size_bytes(0), buffer_size_bytes2(0) {}

    // Constructor
    sub_domain(sycl::queue &q, size_t nx, size_t ny, size_t nz, size_t max_elements) : queue(q), nb_x(nx), nb_y(ny), nb_z(nz), max_device_elements(max_elements) {
        x_stride = nb_x + 2 * ghost_size;
        y_stride = nb_y + 2 * ghost_size;
        z_stride = nb_z + 2 * ghost_size;
        zy_stride = z_stride * y_stride;
        domain_size = x_stride * zy_stride;
        buffer_size_bytes = 2 * domain_size * sizeof(DATATYPE);  // Size for actual data transfers
        buffer_size_bytes2 = 3 * domain_size * sizeof(DATATYPE); // Size for actual data transfers
        init();
    }

    // Destructor
    ~sub_domain() { cleanup(); }

    // Disable copy
    sub_domain(const sub_domain &) = delete;
    sub_domain &operator=(const sub_domain &) = delete;

    // Move constructor: defaulted
    sub_domain(sub_domain &&other) noexcept = default;
    sub_domain &operator=(sub_domain &&other) noexcept = delete;

    private:
    void init() {
        // Host buffers: allocate only what we need (actual domain size)
        h_rhoE = vector<DATATYPE>(2 * domain_size);
        h_uvw = vector<DATATYPE>(3 * domain_size);

        // Device buffers: allocate max_device_elements for alignment/performance benefits
        // We won't use all of this allocation, but it helps with memory alignment
        d_rhoE      = sycl::malloc_device<DATATYPE>(2 * max_device_elements, queue);
        d_uvw       = sycl::malloc_device<DATATYPE>(3 * max_device_elements, queue);
        d_rhoE_next = sycl::malloc_device<DATATYPE>(2 * max_device_elements, queue);
        d_uvw_next  = sycl::malloc_device<DATATYPE>(3 * max_device_elements, queue);
        Dt_next     = sycl::malloc_device<DATATYPE>(1, queue);

        if (!d_rhoE || !d_uvw || !d_rhoE_next || !d_uvw_next || !Dt_next) {
            throw std::bad_alloc();
        }
    }

    void cleanup() {
        if (d_rhoE) { sycl::free(d_rhoE, queue); d_rhoE = nullptr; }
        if (d_uvw) { sycl::free(d_uvw, queue); d_uvw = nullptr; }
        if (d_rhoE_next) { sycl::free(d_rhoE_next, queue); d_rhoE_next = nullptr; }
        if (d_uvw_next) { sycl::free(d_uvw_next, queue); d_uvw_next = nullptr; }
        if (Dt_next) { sycl::free(Dt_next, queue); Dt_next = nullptr; }
    }
};

/* Helper function : print usage */
static int print_usage(char *exec)
{
    printf("Hydrodynamics 3D\n");
    printf("Usage : %s [-xyztiowh]\n", exec);
    printf("\n");
    printf("Options : \n"
        " -j Set the number of sub-domains in the X axis. (--sdx) Default : %ld\n"
        " -k Set the number of sub-domains in the Y axis. (--sdy) Default : %ld\n"
        " -l Set the number of sub-domains in the Y axis. (--sdz) Default : %ld\n"
        " -x Set the number of spatial grid points in the X axis. Default : %ld\n"
        " -y Set the number of spatial grid points in the Y axis. Default : %ld\n"
        " -z Set the number of spatial grid points in the Z axis. Default : %ld\n"
        " -t Set the time at which the simulation stops (does not translate to iterations count). Default : %.2e\n"
        " -i Set the max number of iterations. First of iterations or timestep is reached stops the simulation. Default : no limitation\n"
        " -o Set the output filename. Default : %s\n"
        " -w Write result data each <w> timestep. Default : only write final result"
        "\n"
        " -h, --help Show this message and exit\n",
        NB_SUBDOMAINS_X, NB_SUBDOMAINS_Y, NB_SUBDOMAINS_Z, NB_X, NB_Y, NB_Z, MAX_T, out_filename.c_str());
    return 0;
}

static void write_results(const vector<vector<vector<sub_domain>>> &subdomains, const size_t iteration,
                          const double timestep)
{
    namespace fs = std::filesystem;

    stringstream ss;
    ss << out_filename << "_" << iteration << "_" << std::fixed << std::setprecision(4) << timestep << ".csv";
    std::string filename = ss.str();

    stringstream buffer;

    for (size_t sd_i = 0; sd_i < NB_SUBDOMAINS_X; ++sd_i) {
        for (size_t sd_j = 0; sd_j < NB_SUBDOMAINS_Y; ++sd_j) {
            for (size_t sd_k = 0; sd_k < NB_SUBDOMAINS_Z; ++sd_k) {
                size_t sd_index = sd_i * (NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z) + sd_j * NB_SUBDOMAINS_Z + sd_k;
                if (rank != sd_index % nprocs) continue;
                const sub_domain &cur = subdomains[sd_i][sd_j][sd_k];

                // get local idx
                for (size_t cur_i = 0; cur_i < cur.nb_x; ++cur_i) {
                    for (size_t cur_j = 0; cur_j < cur.nb_y; ++cur_j) {
                        for (size_t cur_k = 0; cur_k < cur.nb_z; ++cur_k) {
                            // local idx in subdomain .. accounting for ghost cells
                            const size_t local_sd_index = (cur_i + ghost_size) * cur.zy_stride + (cur_j + ghost_size) * cur.z_stride + (cur_k + ghost_size);
                            // global position in the whole domain
                            const size_t i = SUBDOMAIN_SIZE_X * sd_i + cur_i;
                            const size_t j = SUBDOMAIN_SIZE_Y * sd_j + cur_j;
                            const size_t k = SUBDOMAIN_SIZE_Z * sd_k + cur_k;

                            buffer << i << ',' << j << ',' << k << ','
                                   << cur.h_rhoE[2 * local_sd_index] << ','
                                   << cur.h_rhoE[2 * local_sd_index + 1] << ','
                                   << cur.h_uvw[3 * local_sd_index] << ','
                                   << cur.h_uvw[3 * local_sd_index + 1] << ','
                                   << cur.h_uvw[3 * local_sd_index + 2] << '\n';
                        }
                    }
                }
            }
        }
    }

    // Each process write it's subdomain data into final csv file in parallel, no race conditions/interleaving
    std::string local_data = buffer.str();
    MPI_Offset local_size = local_data.size();
    MPI_Offset offset = 0;

    // Compute global byte offset for this rank using an exclusive prefix sum (Exscan)
    // Based on MPI ranks, computing the prefix sum of the local_size and rank's offset to compute next's offset.
    MPI_Exscan(&local_size, &offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) offset = 0;

    const std::string header = "x,y,z,rho,p,u,v,w\n";
    const MPI_Offset header_size = header.size();

    MPI_File fh;

    if (rank == 0) { // Truncating content
        MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        MPI_File_set_size(fh, 0);
        MPI_File_close(&fh);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // Header
    if (rank == 0) MPI_File_write_at(fh, 0, header.c_str(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE);

    offset += header_size;

    // Each rank writes its data at its unique offset (non-overlapping, no race condition or interleaving)
    MPI_File_write_at(fh, offset, local_data.c_str(), local_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
}

static double compute_total_mass(const vector<vector<vector<sub_domain>>> &subdomains)
{
    long double total_mass = 0.0L;
    long double c = 0.0L;

    for (size_t sd_i = 0; sd_i < NB_SUBDOMAINS_X; ++sd_i) {
        for (size_t sd_j = 0; sd_j < NB_SUBDOMAINS_Y; ++sd_j) {
            for (size_t sd_k = 0; sd_k < NB_SUBDOMAINS_Z; ++sd_k) {
                size_t sd_index = sd_i * (NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z) + sd_j * NB_SUBDOMAINS_Z + sd_k;
                if (rank != sd_index % nprocs) continue;
                const sub_domain &cur = subdomains[sd_i][sd_j][sd_k];

                for (size_t cur_i = ghost_size; cur_i < cur.x_stride - ghost_size; ++cur_i) {
                    for (size_t cur_j = ghost_size; cur_j < cur.y_stride - ghost_size; ++cur_j) {
                        for (size_t cur_k = ghost_size; cur_k < cur.z_stride - ghost_size; ++cur_k) {
                            const size_t local_sd_index = cur_i * cur.zy_stride + cur_j * cur.z_stride + cur_k;
                            long double var = static_cast<long double>(cur.h_rhoE[2*local_sd_index]);
                            long double t = total_mass + var;
                            // Kahan summation compensation: improves numerical accuracy by tracking lost low-order bits during floating-point addition.
                            if (fabsl(total_mass) >= var) c += (total_mass - t) + var;
                            else c += (var - t) + total_mass;
                            total_mass = t;
                        }
                    }
                }
            }
        }
    }
    double global_initial_mass;
    double local_initial_mass = static_cast<double>(total_mass + c);
    MPI_Allreduce(&local_initial_mass, &global_initial_mass, 1, MPI_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    return global_initial_mass;
}

static double ensure_mass_conservation(double initial_mass, const vector<vector<vector<sub_domain>>> &subdomains,
                                       const size_t it, const double t)
{
    double current_mass = compute_total_mass(subdomains);
    if (rank == 0) {
        double mass_change = fabs(initial_mass - current_mass) / double(NB_X * NB_Y);
        printf("[it=%4lu] t=%.4e Total density : %.7e, change in mass: %.1e (should be close to 0)\n", it, t,
               current_mass, mass_change);
        return mass_change;
    }
    return 0.0;
}

static void set_init_conditions_diffusion(sub_domain &cur, const size_t base_i, const size_t base_j,
                                          const size_t base_k, const DATATYPE u_inside, const DATATYPE u_outside,
                                          const DATATYPE p_inside, const DATATYPE p_outside,
                                          const DATATYPE default_ux, const DATATYPE default_uy,
                                          const DATATYPE default_uz, DATATYPE &min_Dt, const DATATYPE C,
                                          const DATATYPE min_spacing, const DATATYPE gamma)
{
    // Init condition : a very dense circle 1/4 the size of the simulation at its center
    min_Dt = LITERAL(1e20);
    const DATATYPE gamma_minus_one = gamma - ONE;

    const double center_x = double(NB_X + 1) / 2.0;
    const double center_y = double(NB_Y + 1) / 2.0;
    const double center_z = double(NB_Z + 1) / 2.0;
    const double radius = std::min({ center_x, center_y, center_z }) / 3;

    for (size_t cur_i = 0; cur_i < cur.x_stride; ++cur_i) {
        for (size_t cur_j = 0; cur_j < cur.y_stride; ++cur_j) {
            for (size_t cur_k = 0; cur_k < cur.z_stride; ++cur_k) {
                const size_t i = base_i + cur_i;
                const size_t j = base_j + cur_j;
                const size_t k = base_k + cur_k;
                double distance = sqrt((double(i) - center_x) * (double(i) - center_x) +
                                       (double(j) - center_y) * (double(j) - center_y) +
                                       (double(k) - center_z) * (double(k) - center_z));
                const size_t local_sd_index = cur_i * cur.zy_stride + cur_j * cur.z_stride + cur_k;

                DATATYPE U_rho, inv_Q_rho, U_ux, U_uy, U_uz, U_p;
                DATATYPE Q_rho, Q_ux, Q_uy, Q_uz, Q_p;

                U_rho = (distance <= radius ? u_inside : u_outside);
                U_p   = (distance <= radius ? p_inside : p_outside) / gamma_minus_one;
                U_ux  = default_ux;
                U_uy  = default_uy;
                U_uz  = default_uz;

                // Conservative variables
                cur.h_rhoE[2*local_sd_index]   = U_rho;
                cur.h_rhoE[2*local_sd_index+1] = U_p;
                cur.h_uvw [3*local_sd_index]   = U_ux;
                cur.h_uvw [3*local_sd_index+1] = U_uy;
                cur.h_uvw [3*local_sd_index+2] = U_uz;

                // Compute first Dt
                // Primitive variables
                Q_rho = U_rho;
                inv_Q_rho = ONE / Q_rho;
                Q_ux = U_ux * inv_Q_rho;
                Q_uy = U_uy * inv_Q_rho;
                Q_uz = U_uz * inv_Q_rho;

                // e_int = U_p * inv_Q_rho - HALF * (Q_ux * Q_ux) - HALF * (Q_uy * Q_uy);
                DATATYPE temp_e_int = Q_ux*Q_ux + Q_uy*Q_uy + Q_uz*Q_uz;
                DATATYPE e_int = U_p * inv_Q_rho - HALF * temp_e_int;
                Q_p = gamma_minus_one * Q_rho * e_int;

                // compute speed of sound
                DATATYPE c_s = SQRT(gamma * Q_p * inv_Q_rho);

                // Compute Dt
                DATATYPE Unorm = SQRT(temp_e_int);
                DATATYPE local_Dt = C * min_spacing / (c_s + Unorm);

                min_Dt = std::min(min_Dt, local_Dt);
            }
        }
    }
}

static void update_bound(const sub_domain &from, sub_domain &to, const size_t index_from, const size_t index_to)
{
    to.h_rhoE[2*index_to]   = from.h_rhoE[2*index_from];   // rho
    to.h_rhoE[2*index_to+1] = from.h_rhoE[2*index_from+1]; // E
    to.h_uvw [3*index_to]   = from.h_uvw [3*index_from];   // u
    to.h_uvw [3*index_to+1] = from.h_uvw [3*index_from+1]; // v
    to.h_uvw [3*index_to+2] = from.h_uvw [3*index_from+2]; // w
}

static void update_ghost_cells_single_device(vector<vector<vector<sub_domain>>> &subdomains, it_timers_t &T)
{
    struct timespec boundaries_x_t1, boundaries_x_t2;

    // Periodic X boundaries on a single subdomain
    sub_domain &cur = subdomains[0][0][0];
    const size_t y_stride  = cur.y_stride;
    const size_t z_stride  = cur.z_stride;
    const size_t zy_stride = y_stride * z_stride;
    const size_t copy_size_rhoE = 2*zy_stride * sizeof(DATATYPE);
    const size_t copy_size_uvw  = 3*zy_stride * sizeof(DATATYPE);

    clock_gettime(CLOCK_MONOTONIC, &boundaries_x_t1);

    // Copy last real plane (i = nb_x) to left ghost (i = 0)
    const size_t index_from = cur.nb_x * zy_stride;
    const size_t index_to   = 0;

    std::memcpy(&cur.h_rhoE[2*index_to], &cur.h_rhoE[2*index_from], copy_size_rhoE);
    std::memcpy(&cur.h_uvw [3*index_to], &cur.h_uvw [3*index_from], copy_size_uvw);

    // Copy first real plane (i = 1) to right ghost (i = nb_x + 1)
    const size_t index_from2 = ghost_size*zy_stride;
    const size_t index_to2 = (cur.nb_x + ghost_size)*zy_stride;

    std::memcpy(&cur.h_rhoE[2*index_to2], &cur.h_rhoE[2*index_from2], copy_size_rhoE);
    std::memcpy(&cur.h_uvw [3*index_to2], &cur.h_uvw [3*index_from2], copy_size_uvw);

    clock_gettime(CLOCK_MONOTONIC, &boundaries_x_t2);
    T.boundaries_x += get_time_us(boundaries_x_t1, boundaries_x_t2);
    return;
}

// Dirty Ghost-Cells update : hard-coded copy from 0 to 1, assuming only 2 subdomains on the X axis.
static void update_ghost_cells(vector<vector<vector<sub_domain>>> &subdomains, it_timers_t &T)
{
    struct timespec boundaries_x_t1, boundaries_x_t2;

    const int other_rank = (rank + 1) % nprocs;

    // Loop all subdomains owned by this rank (general), though current config is 2 ranks on X only
    for (size_t sd_i = 0; sd_i < NB_SUBDOMAINS_X; ++sd_i) {
        for (size_t sd_j = 0; sd_j < NB_SUBDOMAINS_Y; ++sd_j) {
            for (size_t sd_k = 0; sd_k < NB_SUBDOMAINS_Z; ++sd_k) {
                size_t sd_index = sd_i * (NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z) + sd_j * NB_SUBDOMAINS_Z + sd_k;
                if (rank != sd_index % nprocs) continue;

                sub_domain &cur = subdomains[sd_i][sd_j][sd_k];

                MPI_Request requests[40];
                int req = 0;

                int send_base_tag = (rank == 0) ? 90100 : 91000;
                int recv_base_tag = (rank == 0) ? 91000 : 90100;
                // Offset tags by subdomain coords to avoid collisions if ever multiple per rank
                int sd_tag_offset = static_cast<int>((sd_j * NB_SUBDOMAINS_Z + sd_k) * 100);

                MPI_Barrier(MPI_COMM_WORLD);
                clock_gettime(CLOCK_MONOTONIC, &boundaries_x_t1);

                // Exchange yz-planes across X between ranks
                const size_t yz_count = cur.zy_stride;
                const size_t idx_from_right = cur.nb_x * cur.zy_stride;                // last interior plane
                const size_t idx_from_left  = ghost_size * cur.zy_stride;              // first interior plane
                const size_t idx_to_left    = 0;                                       // left ghost plane
                const size_t idx_to_right   = (cur.nb_x + ghost_size) * cur.zy_stride; // right ghost plane

                // Right interior -> left ghost (neighbor)
                MPI_Isend(cur.h_rhoE.data() + 2*idx_from_right, 2*yz_count, MPI_DATATYPE, other_rank, send_base_tag + sd_tag_offset + 11, MPI_COMM_WORLD, &requests[req++]);
                MPI_Isend(cur.h_uvw.data()  + 3*idx_from_right, 3*yz_count, MPI_DATATYPE, other_rank, send_base_tag + sd_tag_offset + 12, MPI_COMM_WORLD, &requests[req++]);

                MPI_Irecv(cur.h_rhoE.data() + 2*idx_to_left,    2*yz_count, MPI_DATATYPE, other_rank, recv_base_tag + sd_tag_offset + 11, MPI_COMM_WORLD, &requests[req++]);
                MPI_Irecv(cur.h_uvw.data()  + 3*idx_to_left,    3*yz_count, MPI_DATATYPE, other_rank, recv_base_tag + sd_tag_offset + 12, MPI_COMM_WORLD, &requests[req++]);

                // Left interior -> right ghost (neighbor)
                MPI_Isend(cur.h_rhoE.data() + 2*idx_from_left,  2*yz_count, MPI_DATATYPE, other_rank, send_base_tag + sd_tag_offset + 21, MPI_COMM_WORLD, &requests[req++]);
                MPI_Isend(cur.h_uvw.data()  + 3*idx_from_left,  3*yz_count, MPI_DATATYPE, other_rank, send_base_tag + sd_tag_offset + 22, MPI_COMM_WORLD, &requests[req++]);

                MPI_Irecv(cur.h_rhoE.data() + 2*idx_to_right,   2*yz_count, MPI_DATATYPE, other_rank, recv_base_tag + sd_tag_offset + 21, MPI_COMM_WORLD, &requests[req++]);
                MPI_Irecv(cur.h_uvw.data()  + 3*idx_to_right,   3*yz_count, MPI_DATATYPE, other_rank, recv_base_tag + sd_tag_offset + 22, MPI_COMM_WORLD, &requests[req++]);

                MPI_Waitall(req, requests, MPI_STATUSES_IGNORE);

                MPI_Barrier(MPI_COMM_WORLD);
                clock_gettime(CLOCK_MONOTONIC, &boundaries_x_t2);

                T.boundaries_x += get_time_us(boundaries_x_t1, boundaries_x_t2);
            }
        }
    }
}

int main(int argc, char *argv[])
{
#if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
    auto selector = default_selector_v;
#endif

    /* MPI init */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    single_device_mode = (nprocs == 1);

    if (!single_device_mode && nprocs != 2) {
        if (rank == 0) fprintf(stderr, "Error: require exactly 2 MPI ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Argument parsing */
    const char *short_options = "j:k:l:x:y:z:t:i:o:w:h";
    const struct option long_options[] = { { "sdx", required_argument, nullptr, 'j' },
                                           { "sdy", required_argument, nullptr, 'k' },
                                           { "sdz", required_argument, nullptr, 'l' },
                                           { "x_points", required_argument, nullptr, 'x' },
                                           { "y_points", required_argument, nullptr, 'y' },
                                           { "z_points", required_argument, nullptr, 'z' },
                                           { "max_time", required_argument, nullptr, 't' },
                                           { "max_iterations", required_argument, nullptr, 'i' },
                                           { "output_filename", required_argument, nullptr, 'o' },
                                           { "write_interval", required_argument, nullptr, 'w' },
                                           { "help", no_argument, nullptr, 'h' },
                                           { nullptr, 0, nullptr, 0 } };

    int option;
    while ((option = getopt_long_only(argc, argv, short_options, long_options, nullptr)) != -1) {
        switch (option) {
            case 'h': return print_usage(argv[0]);
            case 'j': NB_SUBDOMAINS_X = size_t(stoi(optarg)); break;
            case 'k': NB_SUBDOMAINS_Y = size_t(stoi(optarg)); break;
            case 'l': NB_SUBDOMAINS_Z = size_t(stoi(optarg)); break;
            case 'x': NB_X = size_t(stoi(optarg)); break;
            case 'y': NB_Y = size_t(stoi(optarg)); break;
            case 'z': NB_Z = size_t(stoi(optarg)); break;
            case 't': MAX_T = atof(optarg); break;
            case 'i': MAX_IT = size_t(stoi(optarg)); break;
            case 'o': out_filename = string(optarg); break;
            case 'w': write_interval = size_t(stoi(optarg)); break;
            default: cerr << "Unknown option\n"; return 1;
        }
    }

    if (single_device_mode) { NB_SUBDOMAINS_X = 1; NB_SUBDOMAINS_Y = 1; }

    /* Code */
    const DATATYPE DX = ONE / static_cast<DATATYPE>(NB_X + 1);
    const DATATYPE DY = ONE / static_cast<DATATYPE>(NB_Y + 1);
    const DATATYPE DZ = ONE / static_cast<DATATYPE>(NB_Z + 1);
    const DATATYPE min_spacing = std::min({ DX, DY, DZ });

    size_t needed_cache_size;

    std::ostringstream buffer;

    if (rank == 0) {
        buffer << "Configuration : \n"
               << "  Using Finite Volume method\n"
               << "  Spatial Points X axis (nb_x)  : " << NB_X << "\n"
               << "  Spatial Points Y axis (nb_y)  : " << NB_Y << "\n"
               << "  Spatial Points Z axis (nb_z)  : " << NB_Z << "\n"
               << "  Max Timestep (max_t)          : " << MAX_T << "\n"
               << "  Max iterations (max_it)       : " << MAX_IT << "\n"
               << "  Step in x (dx)                : " << DX << "\n"
               << "  Step in y (dy)                : " << DY << "\n"
               << "  Step in z (dz)                : " << DZ << "\n";

        if (write_interval) buffer << "  Write interval                : " << write_interval << "\n";

        // Check cache size
        needed_cache_size = 2 * (NB_Y + 2 * ghost_size) * (NB_Z + 2 * ghost_size) + 1;
        buffer << "\n"
               << "  Device Max cache size (2*nb_y*nb_z)           : " << CACHE_SIZE << "\n"
               << "  Used cache size                               : " << needed_cache_size << "\n";
        cerr << buffer.str();
        buffer.str(""); buffer.clear();

        if (needed_cache_size > CACHE_SIZE) {
            buffer << "\nThis configuration can't run on FPGA : needed cache size exceeds the available one\n"
                   << " Max available : " << CACHE_SIZE        << " of 2 * (NB_Y + 2) * (NB_Z + 2) + 1\n"
                   << " Requested     : " << needed_cache_size << " of 2 * (" << NB_Y << " + 2) * (" << NB_Z << " + 2) + 1\n";
            cerr << buffer.str();
            exit(1);
        }
    }

    vector<vector<vector<sub_domain>>> subdomains; // Rank 0 only
    vector<it_timers_t> timers;                    // Rank 0 only

    printf("[rank %d] Creating device queue - loading FPGA design\n", rank);

    auto t1 = high_resolution_clock::now();
    sycl::queue queue(selector);
    auto t2 = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t_queue = t2 - t1;

    buffer << "\n[rank " << rank << "] Queue created in " << std::fixed << std::setprecision(2) << (t_queue.count() / 1e3) << "s \n";

    auto device = queue.get_device();

    // Default
    size_t max_block_size = device.get_info<info::device::max_work_group_size>();
    size_t max_EU_count = device.get_info<info::device::max_compute_units>();

    // Mem
    constexpr size_t one_gb = size_t(1ULL << 30); // 1073741824
    size_t global_mem_b = device.get_info<info::device::global_mem_size>();
    double global_mem_gb = double(global_mem_b) / double(one_gb);
    size_t base_align = device.get_info<sycl::info::device::mem_base_addr_align>();     // in bits
    size_t page_sz = device.get_info<sycl::info::device::global_mem_cache_line_size>(); // in bytes
    // size_t bytes_align = base_align / 8;

    // Requirements
    constexpr size_t arrays_count = 2 * NBVAR;
    const size_t domain_size = (NB_X + 2 * ghost_size) * (NB_Y + 2 * ghost_size) * (NB_Z + 2 * ghost_size);
    const size_t bytes_needed = sizeof(DATATYPE) * arrays_count * domain_size;
    const double needed_mem_gb = double(bytes_needed) / double(one_gb) / nprocs;
    // size_t max_problem_size = size_t(sqrt(double(global_mem_b) / double(sizeof(DATATYPE) * arrays_count)));

#if FPGA_HARDWARE
    // Safety offset used when estimating the maximum device elements to allocate.
    constexpr size_t DEVICE_OFFSET = 8095;
    size_t max_device_elements = global_mem_b / arrays_count / sizeof(DATATYPE) - DEVICE_OFFSET; // - k
#else
    size_t max_device_elements = domain_size;
#endif

    buffer << "[rank " << rank << "] Device: " << device.get_info<info::device::name>() << "\n"
           << "[rank " << rank << "] Max Work Group Size        : " << max_block_size << "\n"
           << "[rank " << rank << "] Max EUCount                : " << max_EU_count << "\n"
           << "[rank " << rank << "] Required Alignment         : " << base_align << " bits\n"
           << "[rank " << rank << "] Cache Line Size            : " << page_sz << " bytes\n"
           << "[rank " << rank << "] Max Global Memory          : " << std::fixed << std::setprecision(2)
           << global_mem_gb << " GB\n"
           << "[rank " << rank << "] Estimated Required Mem     : " << std::fixed << std::setprecision(2)
           << needed_mem_gb << " GB\n"
        //    << "[rank " << rank << "] Max problem size (Max - k) : " << std::fixed
        //    << std::setprecision(2) << max_device_elements << "\n"
        //    << "[rank " << rank << "] Max problem size        : " << std::fixed << std::setprecision(2) << max_problem_size << " GB\n"

#if FPGA_HARDWARE
           << "[rank " << rank << "] Allocation Strategy        : Maximum (FPGA alignment)\n"
#else
           << "[rank " << rank << "] Allocation Strategy        : Minimum (CPU/EMULATOR efficiency)\n"
#endif
           << "[rank " << rank << "] Max device elements        : " << std::fixed << std::setprecision(2)
           << max_device_elements << "\n"

           << "\n";

    cerr << buffer.str();
    buffer.str(""); buffer.clear();

    // Check Mem
    int local_mem_ok = (needed_mem_gb <= global_mem_gb) ? 1 : 0;

    int global_mem_ok = 0;
    MPI_Allreduce(&local_mem_ok, &global_mem_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (!global_mem_ok){
        if (needed_mem_gb > global_mem_gb) cerr << "[rank " << rank << "] Insufficient device memory. Needed: " << needed_mem_gb << " GB, Available: " << global_mem_gb << " GB\n";
        MPI_Finalize();
        return 1;
    }

    // Init condition : a 100 times higher pressurized circle 1/4 the size of the simulation at its center
    constexpr DATATYPE u_inside = LITERAL(1.0), u_outside = LITERAL(1.2);
    constexpr DATATYPE p_inside = LITERAL(10.0), p_outside = LITERAL(0.1);
    constexpr DATATYPE default_ux = ZERO; // No initial x-velocity
    constexpr DATATYPE default_uy = ZERO; // No initial y-velocity
    constexpr DATATYPE default_uz = ZERO; // No initial z-velocity

    // Variable defines
    constexpr DATATYPE C = LITERAL(0.5);     // Hard define CFL condition to limit numeric scheme instabilities
    constexpr DATATYPE gamma = LITERAL(1.4); // Heat capacity Ratio
    constexpr DATATYPE K = LITERAL(1.1);     // Safety factor

    constexpr DATATYPE gamma_minus_one = gamma - ONE;
    constexpr DATATYPE divgamma = ONE / gamma_minus_one;

    // Global scope within main/simulation function
    DATATYPE Dt_local;                // Host-side, used by all ranks

    it_timers_t *T = nullptr;         // Ref to timers.back() - Only used on rank 0
    struct timespec simu_t1, simu_t2; // Only used on rank 0

    double initial_mass;              // Only used on rank 0
    double final_mass_change;         // Rank 0 only

    DATATYPE first_Dt = 1e20;

    // Create a 3D grid of subdomains - INITIALISATION
    SUBDOMAIN_SIZE_X = NB_X / NB_SUBDOMAINS_X;
    SUBDOMAIN_SIZE_Y = NB_Y / NB_SUBDOMAINS_Y;
    SUBDOMAIN_SIZE_Z = NB_Z / NB_SUBDOMAINS_Z;

    for (size_t sd_i = 0; sd_i < NB_SUBDOMAINS_X; ++sd_i) {
        vector<vector<sub_domain>> plane;
        plane.reserve(NB_SUBDOMAINS_Y);
        for (size_t sd_j = 0; sd_j < NB_SUBDOMAINS_Y; ++sd_j) {
            vector<sub_domain> line;
            line.reserve(NB_SUBDOMAINS_Z);
            for (size_t sd_k = 0; sd_k < NB_SUBDOMAINS_Z; ++sd_k) {
                size_t sd_index = sd_i * (NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z) + sd_j * NB_SUBDOMAINS_Z + sd_k;
                if (rank != sd_index % nprocs) { line.emplace_back(queue); continue; }

                size_t nb_x = SUBDOMAIN_SIZE_X;
                size_t nb_y = SUBDOMAIN_SIZE_Y;
                size_t nb_z = SUBDOMAIN_SIZE_Z;
                if (sd_i == NB_SUBDOMAINS_X - 1) nb_x += NB_X % NB_SUBDOMAINS_X;
                if (sd_j == NB_SUBDOMAINS_Y - 1) nb_y += NB_Y % NB_SUBDOMAINS_Y;
                if (sd_k == NB_SUBDOMAINS_Z - 1) nb_z += NB_Z % NB_SUBDOMAINS_Z;

                line.emplace_back(queue, nb_x, nb_y, nb_z, max_device_elements);
                sub_domain &cur = line.back();

                // Initialize initial conditions
                const size_t base_i = SUBDOMAIN_SIZE_X * sd_i;
                const size_t base_j = SUBDOMAIN_SIZE_Y * sd_j;
                const size_t base_k = SUBDOMAIN_SIZE_Z * sd_k;
                DATATYPE min_Dt;

                fprintf(stderr, "Init device arrays ...       ");
                set_init_conditions_diffusion(cur, base_i, base_j, base_k, u_inside, u_outside, p_inside, p_outside,
                                              default_ux, default_uy, default_uz, min_Dt, C, min_spacing, gamma);
                fprintf(stderr, "done\n");

                queue.memcpy(cur.d_rhoE, cur.h_rhoE.data(), cur.buffer_size_bytes);
                queue.memcpy(cur.d_uvw, cur.h_uvw.data(), cur.buffer_size_bytes2);
                queue.wait();

                first_Dt = std::min(first_Dt, min_Dt);
            }
            plane.push_back(std::move(line));
        }
        subdomains.push_back(std::move(plane));
    }

    // Gather all first_Dt, get the minimum and set it to Dt_local
    MPI_Allreduce(&first_Dt, &Dt_local, 1, MPI_DATATYPE, MPI_MIN, MPI_COMM_WORLD);

    initial_mass = compute_total_mass(subdomains);

    if (rank == 0) {
        buffer << "\nInit conditions : Diffusion of a 100 times higher pressurized circle at simulation center\n"
               << "  Default density      : " << u_inside << " inside, " << u_outside << " outside\n"
               << "  Default pression     : " << p_inside << " inside, " << p_outside << " outside\n"
               << "  Default u_x velocity : " << default_ux << "\n"
               << "  Default u_y velocity : " << default_uy << "\n";

        cerr << buffer.str();

        printf("\nInitial total mass       : %.7e\n\n", initial_mass);
        printf("Initial Dt : %.2e\n", Dt_local);
    }

    // Output
    // Each MPI process write it's chunk of subdomains.
    if (write_interval) write_results(subdomains, 0, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &simu_t1);
    MPI_Barrier(MPI_COMM_WORLD);

    //
    double t = 0.0;
    size_t it = 0;

    while (t < MAX_T && it < MAX_IT) {
        timers.emplace_back(it_timers_t{});
        T = &timers.back();

        struct timespec host_to_device_t1, host_to_device_t2;
        struct timespec device_to_host_t1, device_to_host_t2;
        struct timespec total_usage_t1, total_usage_t2;

        if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &total_usage_t1);

        // Update ghost cells before kernel launch
        if (single_device_mode) update_ghost_cells_single_device(subdomains, timers.back());
        else update_ghost_cells(subdomains, timers.back());

        const DATATYPE DtDx = Dt_local / DX;
        const DATATYPE DtDy = Dt_local / DY;
        const DATATYPE DtDz = Dt_local / DZ;

        Dt_local = LITERAL(1e20); // Reset Dt_local for next Dt computation
        DATATYPE Dt_local_min = Dt_local;

        // printf("iteration #%ld - starting launcher ...\n", it);

        for (size_t sd_i = 0; sd_i < NB_SUBDOMAINS_X; ++sd_i) {
            for (size_t sd_j = 0; sd_j < NB_SUBDOMAINS_Y; ++sd_j) {
                for (size_t sd_k = 0; sd_k < NB_SUBDOMAINS_Z; ++sd_k) {
                    size_t sd_index = sd_i * (NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z) + sd_j * NB_SUBDOMAINS_Z + sd_k;
                    if (rank != sd_index % nprocs) continue;

                    sub_domain &cur = subdomains[sd_i][sd_j][sd_k];

                    // fprintf(stderr, "Copy host to device ...");
                    clock_gettime(CLOCK_MONOTONIC, &host_to_device_t1);
                    const size_t idx_from_left  = 0 * cur.y_stride; // left ghost (i=0)
                    const size_t idx_from_right = (cur.nb_x + ghost_size) * cur.zy_stride;  // right ghost (i=nb_x+1)
                    const size_t ghost_plane_bytes = cur.zy_stride * sizeof(DATATYPE);      // plane size, 2 or 3 variables per element (rhoE or uvw)

                    queue.memcpy(cur.d_rhoE + 2*idx_from_left, cur.h_rhoE.data()  + 2*idx_from_left,  2*ghost_plane_bytes);
                    queue.memcpy(cur.d_uvw  + 3*idx_from_left, cur.h_uvw.data()   + 3*idx_from_left,  3*ghost_plane_bytes);
                    queue.memcpy(cur.d_rhoE + 2*idx_from_right, cur.h_rhoE.data() + 2*idx_from_right, 2*ghost_plane_bytes);
                    queue.memcpy(cur.d_uvw  + 3*idx_from_right, cur.h_uvw.data()  + 3*idx_from_right, 3*ghost_plane_bytes);
                    queue.wait();
                    clock_gettime(CLOCK_MONOTONIC, &host_to_device_t2);
                    // fprintf(stderr, " done \n");

                    printf("[rank %d] iteration #%ld - starting launcher [%lu][%lu][%lu] ...\n", rank, it, sd_i, sd_j, sd_k);
                    double fpga_hydro_compute =
                        launcher(cur.d_rhoE, cur.d_uvw, cur.d_rhoE_next, cur.d_uvw_next, cur.Dt_next, C, gamma,
                                 gamma_minus_one, divgamma, K, cur.nb_x, cur.nb_y, cur.nb_z, DtDx, DtDy, DtDz,
                                 min_spacing, queue);
                    queue.wait();
                    printf("[rank %d] done\n", rank);

                    std::swap(cur.d_rhoE, cur.d_rhoE_next);
                    std::swap(cur.d_uvw, cur.d_uvw_next);
                    queue.wait();

                    // Next Dt
                    DATATYPE Dt_local_temp;
                    queue.memcpy(&Dt_local_temp, cur.Dt_next, sizeof(DATATYPE)).wait();
                    Dt_local_min = std::min(Dt_local_min, Dt_local_temp);

                    // FPGA to CPU
                    // fprintf(stderr, "Copy device to host ...");
                    clock_gettime(CLOCK_MONOTONIC, &device_to_host_t1);

                    // Update indexes left and right
                    const size_t idx_real_first = 1 * cur.zy_stride;                  // first real (i=1)
                    const size_t idx_real_last = cur.nb_x * cur.zy_stride;            // last real  (i=nb_x)
                    const size_t real_plane_bytes = cur.zy_stride * sizeof(DATATYPE); // plane size, 2 or 3 variables
                    // per element (rhoE or uvw)

                    queue.memcpy(cur.h_rhoE.data() + 2*idx_real_first, cur.d_rhoE + 2*idx_real_first, 2*real_plane_bytes);
                    queue.memcpy(cur.h_uvw.data()  + 3*idx_real_first, cur.d_uvw  + 3*idx_real_first, 3*real_plane_bytes);

                    queue.memcpy(cur.h_rhoE.data() + 2*idx_real_last, cur.d_rhoE  + 2*idx_real_last,   2*real_plane_bytes);
                    queue.memcpy(cur.h_uvw.data()  + 3*idx_real_last, cur.d_uvw   + 3*idx_real_last,   3*real_plane_bytes);
                    queue.wait();
                    clock_gettime(CLOCK_MONOTONIC, &device_to_host_t2);
                    // fprintf(stderr, " done \n");

                    T->total_compute  += fpga_hydro_compute;
                    // T->total_compute2  += get_time_us(device_computation_t1, device_computation_t2);
                    T->host_to_device += get_time_us(host_to_device_t1, host_to_device_t2);
                    T->device_to_host += get_time_us(device_to_host_t1, device_to_host_t2);
                }
            }
        }

        // Gather Every T.total_compute, T.host_to_device, T.device_to_host, T.boundaries_x, T.boundaries_y Sum them all to rank(0) =>
        // T.total_compute, T.host_to_device, T.device_to_host, T.boundaries_x, T.boundaries_y
        it_timers_t T_total;
        static_assert(sizeof(it_timers_t) == sizeof(double) * 8, "Timers struct must be packed with 7 doubles.");
        MPI_Reduce(T, &T_total, sizeof(it_timers_t) / sizeof(double), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        *T = T_total;

        MPI_Allreduce(&Dt_local_min, &Dt_local, 1, MPI_DATATYPE, MPI_MIN, MPI_COMM_WORLD);

        // Wait every process has finished
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &total_usage_t2);
            T->total_usage = get_time_us(total_usage_t1, total_usage_t2);
        }

        if (write_interval && it && (it % write_interval == 0) && it < MAX_IT && t < MAX_T) {
            for (size_t sd_i = 0; sd_i < NB_SUBDOMAINS_X; ++sd_i) {
                for (size_t sd_j = 0; sd_j < NB_SUBDOMAINS_Y; ++sd_j) {
                    for (size_t sd_k = 0; sd_k < NB_SUBDOMAINS_Z; ++sd_k) {
                        size_t sd_index = sd_i * (NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z) + sd_j * NB_SUBDOMAINS_Z + sd_k;
                        if (rank != sd_index % nprocs) continue;

                        sub_domain &cur = subdomains[sd_i][sd_j][sd_k];

                        // Copy full buffer back to host for output
                        queue.memcpy(cur.h_rhoE.data(), cur.d_rhoE, cur.buffer_size_bytes);
                        queue.memcpy(cur.h_uvw.data() , cur.d_uvw , cur.buffer_size_bytes2);
                        queue.wait();
                    }
                }
            }

            ensure_mass_conservation(initial_mass, subdomains, it, t);
            write_results(subdomains, it, t);
        }

        if (rank == 0) printf("[it=%4lu] hydro_fvm compute time t=%.4e (Dt=%.3e) : %.5f s (%.2f ms, %.0f us, %.0f us, %.0f us, %.2f ms)\n",
                   it, t, Dt_local, T->total_usage / 1e6, T->host_to_device / 1e3, T->boundaries_y,
                   T->boundaries_x, T->total_compute, T->device_to_host / 1e3);

        ++it;
        t += Dt_local;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &simu_t2);

    for (size_t sd_i = 0; sd_i < NB_SUBDOMAINS_X; ++sd_i) {
        for (size_t sd_j = 0; sd_j < NB_SUBDOMAINS_Y; ++sd_j) {
            for (size_t sd_k = 0; sd_k < NB_SUBDOMAINS_Z; ++sd_k) {
                size_t sd_index = sd_i * (NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z) + sd_j * NB_SUBDOMAINS_Z + sd_k;
                if (rank != sd_index % nprocs) continue;

                sub_domain &cur = subdomains[sd_i][sd_j][sd_k];

                queue.memcpy(cur.h_rhoE.data(), cur.d_rhoE, cur.buffer_size_bytes);
                queue.memcpy(cur.h_uvw.data(), cur.d_uvw, cur.buffer_size_bytes2);
                queue.wait();
            }
        }
    }

    final_mass_change = ensure_mass_conservation(initial_mass, subdomains, it, t);
    write_results(subdomains, it, t);

    if (rank == 0) {
        double t_simu_sec = get_time_us(simu_t1, simu_t2) / 1e6;

        // Report II and bandwidth
        // Get frequency from executable file (using aocl info). If not found, assume it's 480MHz
        double frequency = 480e6;
        read_frequency(argv[0], &frequency);

        // Timer statistics and output
        std::cout << "\n";
        stats_t res_host_to_device = print_stats(timers, &it_timers_t::host_to_device, "CPU→FPGA");
        stats_t res_device_to_host = print_stats(timers, &it_timers_t::device_to_host, "FPGA→CPU");
        stats_t res_total_usage    = print_stats(timers, &it_timers_t::total_usage, "TOTAL usage");
        stats_t res_total_compute  = print_stats(timers, &it_timers_t::total_compute, "TOTAL compute");
        stats_t res_boundaries_x   = print_stats(timers, &it_timers_t::boundaries_x, "boundaries X");
        stats_t res_boundaries_y   = print_stats(timers, &it_timers_t::boundaries_y, "boundaries Y");
        stats_t res_boundaries_z   = print_stats(timers, &it_timers_t::boundaries_z, "boundaries Z");

        printf("Total execution time: %.3f s\n", t_simu_sec);

        printf("\n=== Estimated II per kernel\n");
        printf("Assuming Design Frequency is %.2lf MHz\n", frequency / 1e6);

        const double problem_size = double(NB_X*NB_Y*NB_Z);
        const unsigned long boundary_problem_size_x = 2*NB_Y*NB_Z;
        const unsigned long boundary_problem_size_y = 2*NB_X*NB_Z;
        const unsigned long boundary_problem_size_z = 2*NB_X*NB_Y;
        printf("update_boundaries_x   ~ II = %.3lf\n", ii(res_boundaries_x.mean, frequency, double(boundary_problem_size_x)));
        printf("update_boundaries_y   ~ II = %.3lf\n", ii(res_boundaries_y.mean, frequency, double(boundary_problem_size_y)));
        printf("update_boundaries_z   ~ II = %.3lf\n", ii(res_boundaries_z.mean, frequency, double(boundary_problem_size_z)));
        printf("compute_hydro         ~ II = %.3lf\n", ii(res_total_compute.mean, frequency, problem_size));

        printf("TOTAL usage           ~ II = %.3lf\n", ii(res_total_usage.mean, frequency, problem_size));

        printf("\nTiming Breakdown\n");
        double pt = (1 / res_total_usage.mean) * 100;
        printf(" 100 %% (%.2lf s) per iteration\n", res_total_usage.mean / 1e6);
        printf(" %5.1lf %% (%.2lf s) Compute\n", res_total_compute.mean * pt, res_total_compute.mean / 1e6);
        double copies_total_s = res_host_to_device.mean + res_device_to_host.mean;
        printf(" %5.1lf %% (%.2lf s) Copies (%2.1lf %% CPU→FPGA + %2.1lf %% FPGA→CPU) (%.2lf s + %.2lf s)\n",
               copies_total_s * pt, copies_total_s / 1e6, res_host_to_device.mean * pt, res_device_to_host.mean * pt,
               res_host_to_device.mean / 1e6, res_device_to_host.mean / 1e6);
        double boundaries_total_s = res_boundaries_x.mean + res_boundaries_y.mean + res_boundaries_z.mean;
        printf(" %5.1lf %% (%.2lf s) Boundary Conditions (%2.1lf %% BC X + %2.1lf %% BC Y + %2.1lf %% BC Z) (%.2lf s + %.2lf s + %.2lf s)\n",
               boundaries_total_s * pt, boundaries_total_s / 1e6, res_boundaries_x.mean * pt,
               res_boundaries_y.mean * pt, res_boundaries_z.mean * pt, res_boundaries_x.mean / 1e6, res_boundaries_y.mean / 1e6, res_boundaries_z.mean / 1e6);

        printf("\n");
        printf("Subdomains : [%lu][%lu][%lu] = %lu\n", NB_SUBDOMAINS_X, NB_SUBDOMAINS_Y, NB_SUBDOMAINS_Z, NB_SUBDOMAINS_X * NB_SUBDOMAINS_Y * NB_SUBDOMAINS_Z);
        printf("Problem size : %.2e\n", problem_size);
        printf("FPGA RAM usage : %.2f GB (%.1f %%)\n", needed_mem_gb, double(needed_mem_gb / global_mem_gb * 100));
        printf("CACHE_SIZE usage : %ld (%.1f %%)\n", needed_cache_size, double(needed_cache_size) / double(CACHE_SIZE) * 100);
        printf("Mass change : %.2e\n", final_mass_change);

        const double total_bytes = arrays_count * problem_size * sizeof(DATATYPE);
        const size_t zy_stride = (NB_Y + 2 * ghost_size) * (NB_Z + 2 * ghost_size);
        const size_t zy_stride_bytes = NBVAR * zy_stride * sizeof(DATATYPE);

        printf("Estimated Throughput : %.3f GB/s (all code)\n", throughput(total_bytes, res_total_usage.mean));
        printf("Estimated Throughput : %.3f GB/s (FPGA compute)\n", throughput(total_bytes, res_total_compute.mean));
        printf("Estimated Throughput : %.3f GB/s (CPU to FPGA)\n", throughput(zy_stride_bytes, res_host_to_device.mean));
        printf("Estimated Throughput : %.3f GB/s (FPGA to CPU)\n", throughput(zy_stride_bytes, res_device_to_host.mean));

        printf("Performance : %.3f Mc/s (all code)\n", performance(problem_size, res_total_usage.mean));
        printf("Performance : %.3f Mc/s (FPGA compute)\n", performance(problem_size, res_total_compute.mean));
    }

    MPI_Finalize();

    return 0;
}
