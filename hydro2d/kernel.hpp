#ifndef KERNEL_H_
#define KERNEL_H_

#include <sycl/sycl.hpp>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
    #include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#ifdef USE_FLOAT

#define DATATYPE float
#define MPI_DATATYPE MPI_FLOAT
#define ZERO 0.0f
#define HALF 0.5f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define SQRT sqrtf
#define LITERAL(X) X##f
#define MMIN std::min
#define MMAX std::max
#define MABS std::abs

#else

#define DATATYPE double
#define MPI_DATATYPE MPI_DOUBLE
#define ZERO 0.0
#define HALF 0.5
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0
#define SQRT sqrt
#define LITERAL(X) X
#define MMIN std::min
#define MMAX std::max
#define MABS std::abs

#endif

constexpr size_t NBVAR{4};
constexpr size_t CACHE_SIZE{65536}; // Must be > to 2x y_stride

#define SIZETNBVAR static_cast<size_t>(NBVAR)
#define SIZETCACHE_SIZE static_cast<size_t>(CACHE_SIZE)

#define TOTAL_CACHE_SIZE (CACHE_SIZE * NBVAR) // Must be > to 2x z_stride * y_stride

constexpr size_t ID{0};
constexpr size_t IU{1};
constexpr size_t IV{2};
constexpr size_t IP{3};

#include <stddef.h>
#include <sys/time.h>

static inline double get_time_us(struct timespec start, struct timespec end)
{
    return static_cast<double>(end.tv_sec - start.tv_sec) * 1e6 +
           static_cast<double>(end.tv_nsec - start.tv_nsec) / 1e3;
}

extern "C" double launcher(DATATYPE *__restrict__ d_rhoE, DATATYPE *__restrict__ d_uv,
                           DATATYPE *__restrict__ d_rhoE_next, DATATYPE *__restrict__ d_uv_next,
                           DATATYPE *__restrict__ Dt_next, const DATATYPE C, const DATATYPE gamma,
                           const DATATYPE gamma_minus_one, const DATATYPE divgamma, const DATATYPE K,
                           const size_t NB_X, const size_t NB_Y, const DATATYPE &DtDx, const DATATYPE &DtDy,
                           const DATATYPE &min_spacing, sycl::queue queue);

#endif // KERNEL_H_
