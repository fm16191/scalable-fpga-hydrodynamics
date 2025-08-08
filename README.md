## Hydrodynamics 2D (SYCL + MPI, FPGA-focused)

A 2D hydrodynamics solver using a first‑order finite volume method with domain decomposition and MPI. The compute kernel is written in SYCL targeting Intel FPGA devices, with a CPU path and an FPGA emulator for development. This repository is optimized for a 2×1 subdomain layout across two MPI ranks and two FPGA devices.

### Key design choices
- **Decomposition along X only (2×1):** Hardcoded for two MPI ranks. Each rank owns one subdomain along X; Y is not decomposed. This simplifies boundary exchanges to copying a single Y‑stride per wall.
- **Ghost cell exchange (X boundaries):** Each step exchanges the right wall of the left subdomain with the left ghost cells of the right subdomain and vice‑versa using non‑blocking MPI calls. Y periodic boundaries are currently disabled in code.
- **Deterministic initial conditions:** A pressurized circular region at the domain center. Timestep is CFL‑limited and recomputed each iteration.
- **FPGA target:** Default board is `IA840F` (changeable via `BOARD_NAME` in `Makefile`).
- **Single‑stream cache design:** Kernel streams i+1 data and caches the i−1..i+1 neighborhood on‑chip to avoid multiple DRAM streams. This requires `2 * y_stride + 1 ≤ CACHE_SIZE`. With `CACHE_SIZE=65000`, the Y dimension is limited to ≈32500 (i.e., domains up to `NB_X × ~32500`).

## Reproducibility checklist
- **Hardware:** 2 FPGA devices (tested with BittWare IA‑840F) or emulator for development; 2 MPI ranks only.
- **Tooling:**
  - Intel oneAPI DPC++/SYCL and Intel MPI (source your environment):
    ```bash
    source /opt/intel/oneapi/setvars.sh
    ```
  - FPGA add‑on for oneAPI and the appropriate Board Support Package (BSP) for your board.
  - BittWare tools for power logging (for IA840F): `bw_card_monitor`.
  - (Optional) Python 3 plotting stack: `plotly`, `kaleido`, `numpy`, `scipy`.
    ```bash
    python3 -m pip install --user plotly kaleido numpy scipy
    ```
- **Build determinism:** The FPGA link step uses a fixed seed (`-Xsseed=2`) in `Makefile` for repeatable builds.
- **Runtime constraints:** Program aborts unless `mpirun -np 2` is used. The code paths and tags assume exactly 2 ranks.
- **Cache constraint:** `CACHE_SIZE` must satisfy `2*(NB_Y + 2) + 1 ≤ CACHE_SIZE` (checked at runtime). Default `CACHE_SIZE=65000`.

## Requirements
- Linux, bash/zsh
- Tested OS (IA840F setup): Rocky Linux 8, kernel `4.18.0-513.24.1.el8_9.x86_64`
- Intel oneAPI (DPC++, Intel MPI)
- For FPGA hardware build: Intel FPGA add‑on for oneAPI and target BSP
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
  Version 2024.1.0

Note: Patches were applied for BittWare's IA840F as requested during installation.
```

## Hardware description
Board: BittWare IA‑840F (Intel Agilex 7; vendor documentation indicates Agilex 7‑M class)

| Component | Specification |
| --- | --- |
| FPGA Model | Intel Agilex 7 F‑Series AGF027 |
| Release Year | Q2 2019 |
| NM technology | Intel’s 10 nm SuperFin |
| Logic Blocks | 912,800 ALMs |
| DSPs / 32b FPUs | 8,528 DSPs |
| Theoretical Peak | 12.8 TFLOPS (SP) |
| On‑Chip SRAM | 35.9 MB |
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
- `hydro.fpga_emu` for emulator
- `hydro.fpga` for hardware

### Host‑only recompilation
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
- `--sdx/-b` number of subdomains in X (keep at 2)
- `--sdy/-n` number of subdomains in Y (keep at 1)
- `-x/-y` grid points in X/Y
- `-t` max simulated time
- `-i` max iterations
- `-w` write interval (iterations); if 0, only final output is written
- `-o` output filename prefix (default `output`)

### Outputs
- CSV files named `<out>_<iter>_<time>.csv` with header `x,y,rho,p,u,v`.
- Timing and performance stats printed on rank 0 at the end.
 - Mass conservation is verified at each write event (`-w/--write_interval`) and at the end of the simulation.

## Energy measurement workflow (IA840F / BittWare)
The script `energy_record.sh` wraps `bw_card_monitor` to record Total Input Power and Total FPGA Power.

### Run a measurement + simulation (hardware)
```bash
it=100; y=20000; x=$((y*2)); \
./energy_record.sh start "record_${it}_${x}_${y}.record" && \
mpirun -np 2 ./hydro.fpga --sdx 2 --sdy 1 -t 2 -i $it -x $x -y $y &> "res_${it}_${x}_${y}.res"; \
./energy_record.sh stop
```

### Plot power log to SVG
```bash
python3 plot_power_consumption.py "record_${it}_${x}_${y}.record"
# or, if you have a `py` alias
py plot_power_consumption.py "record_${it}_${x}_${y}.record"
```
This generates `fpga_power_consumption_record_${it}_${x}_${y}.svg`.

Notes:
- Power metrics captured: Total FPGA Power (chip power draw) and Total Input Power (entire card).
- Limitation: when using two FPGAs, the script currently logs only the first/default card; extend the script to target both devices.

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
  - Default precision: `double` (changeable via `USE_FLOAT`). We typically expect mass‑change accuracy in the 1e‑14 to 1e‑16 range.

Notes:
- Global memory sufficiency and cache size constraints are checked at startup and printed per rank.
- The hardware build uses `-Xsparallel=8` and `-Xsseed=2` in the link step (see `Makefile`).

## Project structure
- `main.cxx` entry point, MPI orchestration, host/device transfers, I/O, timers
- `kernel.cxx` hydrodynamics FVM kernel and launcher (SYCL single_task)
- `kernel.hpp` numeric types, constants, launcher declaration
- `timers.h` timing utilities and reporting helpers
- `Makefile` build targets for CPU/FPGA‑EMU/FPGA/REPORT; board selection via `BOARD_NAME`
- `energy_record.sh` helper for power logging on BittWare boards
- `plot_power_consumption.py` parser/plotter for `bw_card_monitor` logs

## Known limitations
- Exactly 2 MPI ranks are supported; other layouts are not implemented.
- Decomposition is X‑only; Y periodic boundary code is present but disabled.
- The ghost exchange is specialized for two ranks with linear Y‑stride copies.
- Cache size must satisfy `2*(NB_Y + 2) + 1 ≤ CACHE_SIZE`; otherwise the program exits.
- Scaling to more ranks or 2D decomposition requires generalizing MPI exchanges and boundary handling.

## Citation / Acknowledgments
If you use this code or results in academic work, please cite your repository commit and note the FPGA target (IA840F) and toolchain versions used.

## Host platform configuration (placeholder)
Details such as CPU, memory, NUMA/PCIe topology, BIOS/OS power settings will be documented with performance results.

