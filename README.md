## Hydrodynamics 2D (SYCL + MPI, FPGA-focused)

A 2D hydrodynamics solver using a first-order finite volume method with domain decomposition and MPI. The compute kernel is written in SYCL targeting Intel FPGA devices, with a CPU path and an FPGA emulator for development. This repository is optimized for a 2×1 subdomain layout across two MPI ranks and two FPGA devices.


### Key Design Choices

- **X-only Decomposition (2×1):** Fixed for 2 MPI ranks. Each rank owns one X subdomain. Y remains undecomposed. Simplifies boundary exchange to a single Y-stride per wall.
- **Ghost Cell Exchange (X boundaries):** Non-blocking MPI exchanges of right wall of left subdomain ↔ left ghost cells of right subdomain. Implements periodic X boundaries.
- **Y Boundary Conditions:** Reflective.
- **Deterministic Initial Conditions (Blast Test Case):** Central circular region (1/4 domain size) with 100× higher pressure at domain center. CFL-limited timestep, recomputed each iteration. Parameters:
  - Density is slighty larger outside the circular region (`u_inside = 1.0`, `u_outside = 1.2`)
  - Center is 100x more pressuarized (`p_inside = 10.0`, `p_outside = 0.1`)
  - Default velocities in both x y axis are null. (`default_ux = 0.0`, `default_uy = 0.0`)
  - CFL `C = 0.5`, Heat capacity ratio `gamma = 1.4`, Safety factor `K = 1.1`
- **FPGA Target:** Default board `IA840F` (`BOARD_NAME` adjustable in `Makefile`).
- **Single-Stream Cache Design:** Kernel streams i+1 data, caches i−1..i+1 neighborhood on-chip. Requires `2 * y_stride + 1 ≤ CACHE_SIZE` (hardcoded 65000). Limits Y dimension to ≈ 32500 (`NB_X × ~32500` domains).

## Reproducibility checklist
- **Hardware:** 2 FPGA devices (tested with BittWare IA-840F) or emulator for development; 2 MPI ranks only.
- **Tooling:**
  - Intel oneAPI DPC++/SYCL and Intel MPI (source your environment):
    ```bash
    source /opt/intel/oneapi/setvars.sh
    ```
  - FPGA add-on for oneAPI and the appropriate Board Support Package (BSP) for your board.
  - BittWare tools for power logging (for IA840F): `bw_card_monitor`.
  - Optional Python 3 plotting stack :
    - Power consumption plot : `plotly`, `kaleido`, `numpy`, `scipy`
      ```bash
      python3 -m pip install --user plotly kaleido numpy scipy
      ```
    - 2D Domain Visualization : `pandas`, `matplotlib`, `seaborn`
      ```bash
      python3 -m pip install --user pandas matplotlib seaborn
      ```
- **Build determinism:** The FPGA link step uses a fixed seed (`-Xsseed=2`) in `Makefile` for repeatable builds.
- **Runtime constraints:** Program aborts unless `mpirun -np 2` is used. The code paths and tags assume exactly 2 ranks.
- **Cache constraint:** `NB_Y` must satisfy `2*(NB_Y + 2) + 1 ≤ CACHE_SIZE` (checked at runtime). CACHE_SIZE is hardcoded to 65000 (maximum allowed by design), limiting `NB_Y` for valid results.

## Requirements
- Linux, bash/zsh
- Tested OS (IA840F setup): Rocky Linux 8, kernel `4.18.0-513.24.1.el8_9.x86_64`
- Intel oneAPI (DPC++, Intel MPI)
- For FPGA hardware build: Intel FPGA add-on for oneAPI and target BSP
- For power logging (optional): BittWare `bw_card_monitor`
- (Optional) Python 3 + `plotly`, `kaleido`, `numpy`, `scipy` (for plotting power logs)

### Toolchain versions used (reference)
```text
Compiler:
  $ icpx --version
  Intel(R) oneAPI DPC++/C++ Compiler 2024.1.0 (2024.1.0.20240308)
  Target: x86_64-unknown-linux-gnu
  Thread model: posix
  InstalledDir: /opt/intel_/oneapi/compiler/2024.1/bin/compiler
  Configuration file: /opt/intel_/oneapi/compiler/2024.1/bin/compiler/../icpx.cfg

MPI runtime (from Intel oneAPI):
  $ mpirun --version
  Intel(R) MPI Library for Linux* OS, Version 2021.12 Build 20240213 (id: 4f55822)

FPGA tools:
  $ quartus/bin/quartus_sh --version
  Quartus Prime Shell
  Version 23.1.0 Build 115 03/30/2023 Patches 0.02iofs SC Pro Edition
  Copyright (C) 2023  Intel Corporation. All rights reserved.

BittWare SDK:
  $ bw_card_list --version
  Version 2024.1.0

Note: Patches were applied for BittWare's IA840F as requested during installation.
```

## Hardware description
Board: BittWare IA-840F (Intel Agilex 7-F class)

| Component | Specification |
| --- | --- |
| FPGA Model | Intel Agilex 7 F-Series AGF027 (AGFB027R25A2E2V) |
| Release Year | Q2 2019 |
| NM technology | Intel’s 10 nm SuperFin |
| Logic Blocks | 912,800 ALMs |
| DSPs / 32b FPUs | 8,528 DSPs |
| Theoretical Peak | 12.8 TFLOPS (SP) |
| On-Chip SRAM | 35.9 MB |
| Registers | up to 3,651,200 (≈ 456 KB) |
| DDR4 Memory | 32 GB (2 × 16 GB) |
| Peak Memory Bandwidth | 42.7 GB/s |
| PCIe Bandwidth | 32 GB/s (PCIe 4.0 x16) |
| ASP Version | 2023.1.2 |
| Quartus Version | 23.1 |
| Software Stack | Intel oneAPI 2024.1 |

### Board environment (OFS/BSP)
- Name/version: `ofs_ia840f_shim` (v23.1), platform: `linux64`

## Build

### Switch target board (optional)
Edit `Makefile` and set:
```make
BOARD_NAME := ia840f:ofs_ia840f
```
Adjust to your target if needed.

Common board names:
- `intel_a10gx_pac:pac_a10` - Intel Arria 10
- `intel_s10gx_pac:pac_s10` - Intel Stratix 10
- `ia840f:ofs_ia840f` - BittWare's IA840F card (Intel Agilex 7-F)
- `/path/to/IOFS_BUILD_ROOT/oneapi-asp/<folder>:<variant>` - site-specific IOFS builds

Edit the `Makefile` and uncomment/modify the appropriate board configuration for your system.

### Precision (optional)
Add `USE_FLOAT=1` to build with `float` instead of `double`.

### CPU build (optional)
```bash
make -Bj cpu
```

### FPGA emulator build
```bash
make -Bj fpga_emu
```

### FPGA hardware build
This performs hardware linking and can take a long time.
```bash
make -Bij fpga
```

This produces:
- `hydro.cpu` for CPU
- `hydro.fpga_emu` for emulator
- `hydro.fpga` for hardware

### Host-only recompilation
If the kernel is unchanged, you can relink only the host (seconds vs hours):
```bash
make recompile_fpga
```

## Run

Important: exactly 2 MPI ranks are required. The solver is optimized for a 2×1 layout; while `--sdx/--sdy` options exist, only `--sdx 2 --sdy 1` is supported in practice.

The examples below run up to 200 iterations or until simulated time reaches 2 seconds (not execution time), dump domain state in CSV format every 10 iterations, and use a 60×60 grid.

### Emulator example
```bash
mpirun -np 2 ./hydro.fpga_emu --sdx 2 --sdy 1 -t 2 -i 200 -w 10 -x 60 -y 60
```

### Hardware example
```bash
mpirun -np 2 ./hydro.fpga --sdx 2 --sdy 1 -t 2 -i 200 -w 10 -x 60 -y 60
```

### CLI options
- `--sdx/-j` number of subdomains in X (keep at 2)
- `--sdy/-k` number of subdomains in Y (keep at 1)
- `-x/-y` grid points in X/Y
- `-t` max simulated time
- `-i` max iterations
- `-w` write interval (iterations); if 0, only final output is written
- `-o` output filename prefix (default `output`)

## Outputs
- CSV files: `<out>_<iter>_<time>.csv` with header `x,y,rho,p,u,v`. Can be used for visualization in Paraview or plotting scripts.
- Timing and performance stats printed on rank 0.
- Mass conservation is checked at each write event (`-w/--write_interval`) and at simulation end.

## Energy Measurement Workflow (IA840F / BittWare)
`energy_record.sh` wraps `bw_card_monitor` to record Total Input Power and FPGA Power.

### Run Measurement + Simulation
```bash
it=100; y=20000; x=$((y*2)); \
./energy_record.sh start "record_${it}_${x}_${y}.record"; \
mpirun -np 2 ./hydro.fpga -t 2 -i $it -x $x -y $y &> "res_${it}_${x}_${y}.res"; \
./energy_record.sh stop
```

### Plot Power Log to SVG
```bash
python3 plot_power_consumption.py "record_${it}_${x}_${y}.record"
```

Generates `fpga_power_consumption_record_${it}_${x}_${y}.svg` showing total input and FPGA power over time, with mean power indicated for both.

Notes:
- Power metrics captured: Total FPGA Power (chip power draw) and Total Input Power (entire card).
- Limitation: when using two FPGAs, the script currently logs only the first/default card; extension to target both devices via device IDs is planned.

## Performance metrics (printed on rank 0)
- **Timing breakdown per iteration:**
  - CPU→FPGA copy time
  - Boundary conditions X/Y
  - FPGA compute time
  - FPGA→CPU copy time
  - Total usage time
  - **Estimated Initiation Interval (II):** Derived from mean times and an assumed design frequency (read from executable if present; defaults to 480 MHz).
  - **Throughput (GB/s):** For all code, FPGA compute only, and copy paths.
  - **Performance (Mc/s):** Million cells per second (all code and compute only).
  - **Mass conservation:** Initial mass, final mass change.
  - Default precision: `double` (changeable via `USE_FLOAT`). We typically expect mass-change accuracy in the 1e-14 to 1e-16 range.
- Performance metrics only take into account the FPGA computations and host to device communications within simulation execution. This does not take into account initialisation and domain dumping to CSV files.

Notes:
- Global memory sufficiency and cache size constraints are checked at startup and printed per rank.
- The hardware build uses `-Xsparallel=8` and `-Xsseed=2` in the link step (see `Makefile`).

## Reference run data
For reproducibility verification, the `reference_run` directory contains:
- Sample output CSV files from a benchmark run.
- PNG frame outputs for visual inspection.
- Full execution output logs.
- FPGA synthesis reports (e.g., resource utilization).
- Benchmark scripts with fixed parameters, including full compilation and execution commands.
- A comparison script (`compare_outputs.py`) that computes checksums or diffs across iterations to verify output consistency, supplemented by mass conservation checks.

## Project structure
- `main.cxx` entry point, MPI orchestration, host/device transfers, I/O, timers
- `kernel.cxx` hydrodynamics FVM kernel and launcher (SYCL single_task)
- `kernel.hpp` numeric types, constants, launcher declaration
- `timers.h` timing utilities and reporting helpers
- `Makefile` build targets for CPU/FPGA-EMU/FPGA/REPORT; board selection via `BOARD_NAME`
- `energy_record.sh` helper for power logging on BittWare boards
- `plot_power_consumption.py` parser/plotter for `bw_card_monitor` logs
- `reference_run/` directory for benchmark data, scripts, and verification assets

## Known limitations
- Exactly 2 MPI ranks are supported; other layouts are not implemented.
- Decomposition is X-only; Y periodic boundary code is present but disabled.
- The ghost exchange is specialized for two ranks with linear Y-stride copies.
- Cache size must satisfy `2*(NB_Y + 2) + 1 ≤ CACHE_SIZE`; otherwise the program exits.
- Scaling to more ranks or 2D decomposition requires generalizing MPI exchanges and boundary handling.
- No containerization (e.g., Docker/Singularity) for environment isolation due to time constraints.

## Citation / Acknowledgments
If you use this code or results in academic work, please cite your repository commit and note the FPGA target (IA840F) and toolchain versions used.

## Host platform configuration
- **Model:** Intel® Xeon® Gold 6334 @ 3.60 GHz
- **Sockets:** 2 * 8 cores/socket * 2 threads/core = 32 logical CPUs
- **Base freq:** 3.70 GHz (max 3.70 GHz)
- **Caches:** L1d 48 KB, L1i 32 KB, L2 1.25 MB, L3 18 MB/socket
- **ISA features:** AVX, AVX2, AVX-512 family, FMA, AES, GFNI, VAES, VPCLMULQDQ, SHA-NI, VT-x
- **NUMA nodes:** 2
  - Node 0: CPUs `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30`
  - Node 1: CPUs `1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31`
- **SMT pairs:** Logical pairs `(N, N+16)` are hyperthreads of the same core.
- **RAM:** 256 GB OS-Usable DDR4 ECC Registered, 3200 MT/s, dual-rank modules, 24/32 slots of 16Gb RAM
- **Hugepages:** 2048 * 2 MiB = 4 GiB reserved
- **Kernel:** 4.18.0-513.24.1.el8_9.x86_64
- **Boot params:** `intel_iommu=on pcie=realloc hugepagesz=2M hugepages=2048`
- **CPUfreq:** `intel_cpufreq` driver, governor `performance`, 3.70 GHz fixed, turbo active
- **NUMA policy (runs):** default, physcpubind 0–31, nodebind 0,1, membind 0,1
- **Compiler flags (CPU-relevant):** `-march=native` with Intel oneAPI `icpx`

