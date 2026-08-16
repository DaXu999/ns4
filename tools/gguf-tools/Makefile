CC ?= cc
CXX ?= c++
LLAMA_CPP_DIR ?= ../../../deepseek-v4-quants/llama.cpp
LLAMA_CPP_BUILD ?= $(LLAMA_CPP_DIR)/build-remote
LLAMA_CPP_BIN ?= $(LLAMA_CPP_BUILD)/bin
LLAMA_CPP_CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 \
	-I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include

NATIVE_CPU_FLAG ?= -march=native
CUDA_NVCC := $(shell command -v nvcc 2>/dev/null)
CUDA_HOME ?= $(if $(wildcard /usr/local/cuda/bin/nvcc),/usr/local/cuda,$(if $(CUDA_NVCC),$(patsubst %/bin/nvcc,%,$(CUDA_NVCC)),/usr/local/cuda))
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
QUALITY_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
QUALITY_TARGETS := src/ds4.o src/ds4_cuda.o src/ds4_distributed.o src/ds4_ssd.o src/ds4_layer_pack.o vendor/rax.o src/ds4_gpu_args.o \
	src/cuda/mmq/ds4_ggml_stubs.o src/cuda/mmq/ds4_mmq.o src/cuda/mmq/ds4_mmq_d2r.o \
	src/cuda/mmq/quantize.o src/cuda/mmq/mmid.o src/cuda/mmq/mmvq.o src/cuda/mmq/ds4_repack.o
QUALITY_OBJS := $(addprefix ../../,$(QUALITY_TARGETS))
QUALITY_LINK := $(NVCC) $(NVCCFLAGS) -I../../src -I../../vendor -o
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -ldl -lpthread

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quality-score quality-llama-score

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

quality-score:
	$(MAKE) -C ../.. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)

quality-llama-score:
	$(CXX) $(LLAMA_CPP_CXXFLAGS) -o quality-testing/score_llama quality-testing/score_llama.cpp $(LLAMA_CPP_LDLIBS)

clean:
	rm -f deepseek4-quantize quality-testing/score_official quality-testing/score_llama
