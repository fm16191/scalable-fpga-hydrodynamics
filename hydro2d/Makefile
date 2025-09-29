CCXX := CXX=icpx mpicxx -cxx=icpx -fsycl
CXXFLAGS := -O2 -g -fPIC -march=native -Wall -Wextra #-Weverything
CXXFLAGS += -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-undef -Wno-unused-function -Wno-unsafe-buffer-usage -Wno-global-constructors -Wno-exit-time-destructors
# CXXFLAGS += -Wno-unused-variable
# CXXFLAGS += -O0 -ggdb

ifdef USE_FLOAT
CXXFLAGS += -DUSE_FLOAT
endif

# -fsafe-buffer-usage-suggestions # only in oneAPI 2024.0
FLAGS_FPGA := -fintelfpga

SRC := main.cxx
KERNEL_SRC := kernel.cxx
OBJ := $(SRC:.cxx=.o)
KERNEL_OBJ := $(KERNEL_SRC:.cxx=.o)

BUILD_TYPE :=
OPTION :=

BOARD_NAME := ia840f:ofs_ia840f

TARGET_NAME := hydro
CPU_EXE_NAME := $(TARGET_NAME).cpu
EMU_EXE_NAME := $(TARGET_NAME).fpga_emu
SIMU_EXE_NAME := $(TARGET_NAME).fpga_simu
REPORT_NAME := $(TARGET_NAME)_report.a
FPGA_EXE_NAME := $(TARGET_NAME).fpga

.PHONY: cpu fpga_emu fpga_simu report fpga

%.o: %.cxx
	$(CCXX) $(CXXFLAGS) $(BUILD_TYPE) -c $< -o $@ $(OPTION)

# CPU
cpu: $(OBJ) $(KERNEL_OBJ)
	$(CCXX) -lmpi $(CXXFLAGS) -o $(CPU_EXE_NAME) $^


# Emulator
fpga_emu: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_EMULATOR=1
fpga_emu: $(OBJ) $(KERNEL_OBJ)
	$(CCXX) -lmpi $(BUILD_TYPE) $^ -o $(EMU_EXE_NAME)


# Simulator
fpga_simu: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_SIMULATOR=1 -Xssimulation
fpga_simu: $(OBJ)
	$(CCXX) -lmpi $(BUILD_TYPE) $^ -o $(SIMU_EXE_NAME)

# Report
report: BUILD_TYPE := $(FLAGS_FPGA) -fsycl-link=early -Xshardware -Xstarget=$(BOARD_NAME) -DFPGA_HARDWARE=1 -DREPORT
report: $(KERNEL_SRC)
	$(CCXX) $(CXXFLAGS) $(BUILD_TYPE) $^ -o $(REPORT_NAME)


# Hardware
KERNEL_FPGA_SO := $(KERNEL_SRC:.cxx=.so)
$(KERNEL_FPGA_SO): BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_HARDWARE=1
$(KERNEL_FPGA_SO): $(KERNEL_OBJ)
	$(CCXX) -lmpi $(CXXFLAGS) $(BUILD_TYPE) -shared -Xshardware -Xsparallel=8 -Xsseed=17 -Xstarget=$(BOARD_NAME) -fsycl-link=image $^ -o $@ $(OPTION)

fpga: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_HARDWARE=1
recompile_fpga: BUILD_TYPE := $(FLAGS_FPGA) -DFPGA_HARDWARE=1

fpga: $(OBJ) $(KERNEL_FPGA_SO)
	$(CCXX) -lmpi $(BUILD_TYPE) $^ -o $(FPGA_EXE_NAME) $(OPTION)
recompile_fpga: $(OBJ)
	$(CCXX) -lmpi $(BUILD_TYPE) $^ $(KERNEL_FPGA_SO) -o $(FPGA_EXE_NAME) $(OPTION)

clean:
	rm -rf *.o *.d *.out *.mon *.aocr *.aoco *.cpu *.fpga_emu *.fpga_simu *.a $(FPGA_EXE_NAME)

cleanout:
	rm -f *.csv *.png

cleanres:
	rm -f *.res *.record *.aocx *.usage

cleanall: clean cleanout
