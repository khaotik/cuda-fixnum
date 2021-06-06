CXX ?= g++

INCLUDE_DIRS = -I./include
NVCC_FLAGS = -ccbin $(CXX) -std=c++11 -Xcompiler -Wall,-Wextra
NVCC_OPT_FLAGS = -DNDEBUG
NVCC_TEST_FLAGS = -lineinfo
NVCC_DBG_FLAGS = -g -G
NVCC_LIBS = -lstdc++
NVCC_TEST_LIBS = -lgtest

# TODO.feat what if host has multiple GPUs
.cudaArch:
	@nvcc -o cuda-caps cuda-caps.cu;
	@./cuda-caps > $@
	@rm cuda-caps

all:
	@echo "Please run 'make check' or 'make bench'."

tests/test-suite: tests/test-suite.cu .cudaArch
	$(eval GENCODES := $(shell cat .cudaArch))
	@nvcc $(NVCC_TEST_FLAGS) $(NVCC_FLAGS) -gencode "arch=compute_$(GENCODES),code=sm_$(GENCODES)" $(INCLUDE_DIRS) $(NVCC_LIBS) $(NVCC_TEST_LIBS) -o $@ $<

check: tests/test-suite
	@./tests/test-suite

bench/bench: bench/bench.cu .cudaArch
	$(eval GENCODES := $(shell cat .cudaArch))
	@nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) -gencode "arch=compute_$(GENCODES),code=sm_$(GENCODES)" $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

bench: bench/bench

.PHONY: clean
clean:
	$(RM) tests/test-suite bench/bench .cudaArch
