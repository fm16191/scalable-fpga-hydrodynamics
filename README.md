## Scalable Hydrodynamics on FPGA (SYCL + MPI)

This repository contains a proof-of-concept implementation of compressible hydrodynamics on FPGA accelerators using SYCL (Intel oneAPI) and MPI for domain decomposition across one or two devices. It includes complete 2D and 3D solvers built with the same design principles.

The project supports both single-device and two-device (multi-FPGA via MPI) configurations, periodic/reflective boundary conditions, deterministic initial conditions, and an instrumentation flow for timing, performance, and energy measurements.

This work is made public and reproducible for the SC25 workshop ScalaH 16. See the workshop page: [SCALA workshop 2025 — ScalaH 16](https://www.csm.ornl.gov/srt/conferences/Scala/2025/).


### What’s here
- **2D Hydrodynamics:** SYCL kernel and MPI host orchestrating an X-only decomposition across two ranks and up to two FPGA devices. See `hydro2d/` for details.
- **3D Hydrodynamics:** Mirrors the 2D design (first-order finite volume, MPI domain decomposition, ghost-cell exchange, reflective/periodic boundaries, and identical instrumentation). See `hydro3d/` for details.


## Key features
- **SYCL + FPGA focus:** Single-task kernels with on-chip neighborhood caching for stencil access.
- **MPI domain decomposition:** 1D or 2D decomposition depending on the dimension; current 2D code is X-only (2×1) for two ranks.
- **Boundary conditions:** Periodic (X) and reflective (Y) in 2D; analogous options in 3D.
- **Deterministic initial conditions:** Blast-like test with configurable parameters.
- **Host-side timing and metrics:** Copy/compute timing, throughput, estimated II, and mass conservation checks.
- **Energy measurement workflow:** Optional logging with BittWare tools on IA-840F hardware and plotting scripts.


## Repository layout
- `hydro2d/` — 2D solver
  - `main.cxx` — MPI orchestration, host/device transfers, I/O, timers
  - `kernel.cxx` — SYCL kernel (finite volume update)
  - `kernel.hpp` — Types, constants, launcher decl.
  - `timers.h` — Timing utilities and reporting
  - `Makefile` — Targets for CPU, FPGA emulator, FPGA hardware (board selection)
  - `README.md` — Detailed documentation: design, constraints, build/run, metrics
- `hydro3d/` — 3D solver (same structure as `hydro2d/`)
  - `main.cxx`, `kernel.cxx`, `kernel.hpp`, `timers.h`, `Makefile`, `README.md`
- `scripts/` — Plotting utilities (power log, 2D domain visualization)



## Toolchain and requirements
- Linux, bash/zsh
- Intel oneAPI (DPC++/SYCL and Intel MPI)
- For FPGA hardware builds: Intel FPGA add-on for oneAPI and the appropriate BSP
- Optional (IA-840F): BittWare SDK (`bw_card_monitor`) for power logging
- Optional Python 3 stack for plotting (`plotly`, `kaleido`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`)

Reference toolchain versions are listed in the per-dimension READMEs (e.g., `hydro2d/README.md`). Both solvers target the same stack.


## Build overview

All commands below assume you have sourced oneAPI:
```bash
source /opt/intel/oneapi/setvars.sh
```

### 2D solver
Change directory and choose a board in `Makefile` (`BOARD_NAME`). Common entries include:
- `intel_a10gx_pac:pac_a10` (Arria 10)
- `intel_s10gx_pac:pac_s10` (Stratix 10)
- `ia840f:ofs_ia840f` (BittWare IA-840F)

Build targets in `hydro2d/`:
```bash
make -Bj cpu       # CPU host reference
make -Bj fpga_emu  # FPGA emulator
make -Bij fpga     # FPGA hardware (link step, long)
```

Artifacts:
- `hydro.cpu`, `hydro.fpga_emu`, `hydro.fpga`

Host-only relink (kernel unchanged):
```bash
make recompile_fpga
```

Precision switch:
```bash
make USE_FLOAT=1 <target>
```

### 3D solver
Build targets in `hydro3d/` are identical:
```bash
make -Bj cpu
make -Bj fpga_emu
make -Bij fpga
```
Host-only relink and `USE_FLOAT=1` behave the same.


## Run overview (MPI)

Both solvers are optimized for two ranks on two devices with a 2×1 layout along X.

Emulator:
```bash
cd hydro2d
mpirun -np 2 ./hydro.fpga_emu --sdx 2 --sdy 1 -t 2 -i 200 -w 10 -x 60 -y 60
```

Hardware:
```bash
cd hydro2d
mpirun -np 2 ./hydro.fpga --sdx 2 --sdy 1 -t 2 -i 200 -w 10 -x 60 -y 60
```

3D emulator:
```bash
cd hydro3d
mpirun -np 2 ./hydro3d.fpga_emu --sdx 2 --sdy 1 -t 2 -i 200 -w 10 -x 60 -y 60 -z 60
```

3D hardware:
```bash
cd hydro3d
mpirun -np 2 ./hydro3d.fpga --sdx 2 --sdy 1 -t 2 -i 200 -w 10 -x 60 -y 60 -z 60
```

CLI highlights:
- `--sdx/--sdy` subdomains in X/Y (validated for `--sdx 2 --sdy 1`)
- `-x/-y[/ -z]` grid size per global domain (add `-z` for 3D)
- `-t` max simulated time; `-i` max iterations; `-w` write interval

Outputs:
- 2D: CSV files `<out>_<iter>_<time>.csv` with header `x,y,rho,p,u,v`
- 3D: CSV files `<out>_<iter>_<time>.csv` with header `x,y,z,rho,p,u,v,w`
- Timing/performance statistics on rank 0; mass conservation checks at writes and end


## Domain decomposition and boundaries

2D:
- **Decomposition:** 2×1 across X (two ranks). Ghost-cell exchange on X faces via non-blocking MPI. Y is undecomposed.
- **Boundaries:** Periodic in X; reflective in Y.
- **Kernel cache design:** Single-stream line buffer maintaining i−1..i+1 neighborhood; requires `2*(NB_Y + 2) + 1 ≤ CACHE_SIZE` (default 65000).

3D:
- **Decomposition:** 2×1×1 (or 1×2×1) across one axis for two ranks; ghost exchange generalized across face-neighbor subdomains.
- **Boundaries:** Periodic/reflective per-axis; defaults mirror 2D.
- **Kernel:** Streaming/caching extended to 3D stencils with cache-size validation at startup.


## Reproducibility
- Deterministic initial conditions (blast-like setup)
- Fixed FPGA link seed for repeatable builds (see Makefiles, `-Xsseed=2`)
- Printed environment summary and parameter echo on startup
- Reference toolchain versions documented in `hydro2d/README.md` and `hydro3d/README.md`


## Energy measurement workflow (IA-840F)

Both solvers use a convenience wrapper `energy_record.sh` and plotting scripts under `scripts/`.

Example (hardware):
```bash
cd hydro2d
it=100; y=20000; x=$((y*2)); \
./energy_record.sh start "record_${it}_${x}_${y}.record"; \
mpirun -np 2 ./hydro.fpga -t 2 -i $it -x $x -y $y &> "res_${it}_${x}_${y}.res"; \
./energy_record.sh stop
```

Plot to SVG:
```bash
python3 scripts/plot_power_consumption.py "record_${it}_${x}_${y}.record"
```

Notes:
- Captures Total Input Power and FPGA Power. With two devices, the script presently logs the default/first device; extension to per-device selection is straightforward and planned.


## Performance metrics
Printed on rank 0:
- Copy times (CPU→FPGA, FPGA→CPU), boundary handling, compute time, total
- Estimated Initiation Interval (II) from mean timing and design frequency
- Throughput (GB/s) and performance (Mcells/s) for compute and end-to-end
- Mass conservation delta


## Getting started
1) Ensure oneAPI and (optionally) FPGA add-on are installed and sourced
2) Build `hydro2d/` or `hydro3d/` for your target (`cpu`, `fpga_emu`, or `fpga`)
3) Run with two MPI ranks and 2×1 subdomain layout
4) Inspect CSV outputs and timing logs; optionally record and plot power

For deeper details, constraints, and environment specifics, see the per-dimension READMEs.


## Citation / Acknowledgments
If you use this code or results in academic work, please cite a commit from this repository, and note the FPGA target and toolchain versions (documented in the subproject README). This work accompanies the [SCALA workshop 2025 — ScalaH 16](https://www.csm.ornl.gov/srt/conferences/Scala/2025/).