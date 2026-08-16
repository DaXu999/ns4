CC ?= cc

NATIVE_CPU_FLAG ?= -march=native
SAMPLING_TEST := tests/test_sampling

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

CFLAGS += -D_GNU_SOURCE -fno-finite-math-only -Isrc -Ivendor
CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else ifneq ($(filter sm_121 sm_121a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# Vendored llama.cpp mmq prefill tier (src/cuda/mmq/, see src/cuda/mmq/VENDOR.md).
MMQ_INCLUDES := -Isrc/cuda/mmq
MMQ_OBJS := src/cuda/mmq/ds4_ggml_stubs.o src/cuda/mmq/ds4_mmq.o src/cuda/mmq/ds4_mmq_d2r.o src/cuda/mmq/quantize.o src/cuda/mmq/mmid.o src/cuda/mmq/mmvq.o src/cuda/mmq/ds4_repack.o
CORE_OBJS = src/ds4.o src/ds4_distributed.o src/ds4_ssd.o src/ds4_cuda.o src/ds4_layer_pack.o $(MMQ_OBJS)
CPU_CORE_OBJS = src/ds4_cpu.o src/ds4_distributed.o src/ds4_ssd.o src/ds4_layer_pack.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas

.PHONY: all help clean test test-mxfp4-cuda test-cuda-session-batch test-cuda-mixed-batch dspark-acceptance dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression

all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make test                Build and run tests"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=sm_121

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH="$(CUDA_ARCH)"

ds4: src/ds4_cli.o src/ds4_help.o vendor/linenoise.o src/ds4_gpu_args.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: src/ds4_server.o src/ds4_help.o src/ds4_kvstore.o vendor/rax.o src/ds4_gpu_args.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: src/ds4_bench.o src/ds4_help.o src/ds4_gpu_args.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: src/ds4_eval.o src/ds4_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tools/gguf-tools/quality-testing/score_official.o: tools/gguf-tools/quality-testing/score_official.c src/ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) -Isrc -Ivendor -c -o $@ $<

tools/gguf-tools/quality-testing/score_official: tools/gguf-tools/quality-testing/score_official.o $(CORE_OBJS) vendor/rax.o src/ds4_gpu_args.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: src/ds4_cli_cpu.o src/ds4_server_cpu.o src/ds4_bench_cpu.o src/ds4_eval_cpu.o src/ds4_help.o src/ds4_kvstore.o vendor/linenoise.o vendor/rax.o src/ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 src/ds4_cli_cpu.o src/ds4_help.o vendor/linenoise.o src/ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server src/ds4_server_cpu.o src/ds4_help.o src/ds4_kvstore.o vendor/rax.o src/ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench src/ds4_bench_cpu.o src/ds4_help.o src/ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval src/ds4_eval_cpu.o src/ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda

src/ds4.o: src/ds4.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4.c

src/ds4_ssd.o: src/ds4_ssd.c src/ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_ssd.c

src/ds4_cli.o: src/ds4_cli.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h vendor/linenoise.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_cli.c

src/ds4_distributed.o: src/ds4_distributed.c src/ds4_distributed.h src/ds4.h src/ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_distributed.c

src/ds4_help.o: src/ds4_help.c src/ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_help.c

src/ds4_gpu_args.o: src/ds4_gpu_args.c src/ds4_gpu_args.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_gpu_args.c

src/ds4_server.o: src/ds4_server.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h src/ds4_kvstore.h vendor/rax.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_server.c

src/ds4_bench.o: src/ds4_bench.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_bench.c

src/ds4_eval.o: src/ds4_eval.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_eval.c

src/ds4_kvstore.o: src/ds4_kvstore.c src/ds4_kvstore.h src/ds4.h src/ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ src/ds4_kvstore.c

ds4_test.o: tests/ds4_test.c src/ds4_server.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h src/ds4_kvstore.h vendor/rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c src/ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ tests/cuda_long_context_smoke.c

vendor/rax.o: vendor/rax.c vendor/rax.h vendor/rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ vendor/rax.c

vendor/linenoise.o: vendor/linenoise.c vendor/linenoise.h
	$(CC) $(CFLAGS) -c -o $@ vendor/linenoise.c

src/ds4_cpu.o: src/ds4.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ src/ds4.c

src/ds4_cli_cpu.o: src/ds4_cli.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h vendor/linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/ds4_cli.c

src/ds4_gpu_args_cpu.o: src/ds4_gpu_args.c src/ds4_gpu_args.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/ds4_gpu_args.c

src/ds4_server_cpu.o: src/ds4_server.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h src/ds4_kvstore.h vendor/rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/ds4_server.c

src/ds4_bench_cpu.o: src/ds4_bench.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/ds4_bench.c

src/ds4_eval_cpu.o: src/ds4_eval.c src/ds4.h src/ds4_ssd.h src/ds4_distributed.h src/ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/ds4_eval.c

src/ds4_cuda.o: src/ds4_cuda.cu src/ds4_gpu.h src/ds4_gpu_mgpu.h src/ds4_iq2_tables_cuda.inc src/cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ src/ds4_cuda.cu

# Vendored mmq pieces (see src/cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
src/cuda/mmq/ds4_ggml_stubs.o: src/cuda/mmq/ds4_ggml_stubs.cu src/cuda/mmq/ds4_ggml_stubs.h src/cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

src/cuda/mmq/ds4_mmq.o: src/cuda/mmq/ds4_mmq.cu src/cuda/mmq/ds4_mmq.h src/cuda/mmq/ds4_mmq_d2r.cuh src/cuda/mmq/mmq.cuh src/cuda/mmq/common.cuh src/cuda/mmq/ds4_ggml_stubs.h src/cuda/mmq/quantize.cuh src/cuda/mmq/mmid.cuh src/cuda/mmq/vecdotq.cuh src/cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

src/cuda/mmq/ds4_mmq_d2r.o: src/cuda/mmq/ds4_mmq_d2r.cu src/cuda/mmq/ds4_mmq_d2r.cuh src/cuda/mmq/mmq.cuh src/cuda/mmq/common.cuh src/cuda/mmq/ds4_ggml_stubs.h src/cuda/mmq/vecdotq.cuh src/cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

src/cuda/mmq/quantize.o: src/cuda/mmq/quantize.cu src/cuda/mmq/quantize.cuh src/cuda/mmq/common.cuh src/cuda/mmq/ds4_ggml_stubs.h src/cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

src/cuda/mmq/mmid.o: src/cuda/mmq/mmid.cu src/cuda/mmq/mmid.cuh src/cuda/mmq/common.cuh src/cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

src/cuda/mmq/mmvq.o: src/cuda/mmq/mmvq.cu src/cuda/mmq/mmvq.cuh src/cuda/mmq/common.cuh src/cuda/mmq/ds4_ggml_stubs.h src/cuda/mmq/quantize.cuh src/cuda/mmq/vecdotq.cuh src/cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

src/cuda/mmq/ds4_repack.o: src/cuda/mmq/ds4_repack.cu src/cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o src/ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c src/ds4_layer_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o src/ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c src/ds4_gpu_args.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o src/ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

src/ds4_cpu_test_hooks.o: src/ds4.c src/ds4.h src/ds4_gpu.h src/ds4_gpu_mgpu.h src/ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ src/ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c src/ds4.h src/ds4_gpu_mgpu.h src/ds4_layer_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o src/ds4_cpu_test_hooks.o src/ds4_distributed.o src/ds4_ssd.o src/ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_xdev.o: tests/test_gpu_xdev.c src/ds4_gpu.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o src/ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c src/ds4_gpu.h
	$(CC) $(CFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o src/ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c src/ds4_gpu.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o src/ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

src/ds4_cuda_test_hooks.o: src/ds4.c src/ds4.h src/ds4_gpu.h src/ds4_gpu_mgpu.h src/ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ src/ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c src/ds4.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o src/ds4_gpu_args.o src/ds4_kvstore.o vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c src/ds4.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o src/ds4_cuda_test_hooks.o src/ds4_gpu_args.o src/ds4_kvstore.o vendor/rax.o src/ds4_distributed.o src/ds4_ssd.o src/ds4_cuda.o src/ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c src/ds4.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o src/ds4_gpu_args.o src/ds4_kvstore.o vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_sampling.o: tests/test_sampling.c src/ds4.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -c -o $@ $<

tests/test_sampling: tests/test_sampling.o src/ds4_cuda_test_hooks.o src/ds4_gpu_args.o src/ds4_kvstore.o vendor/rax.o src/ds4_distributed.o src/ds4_ssd.o src/ds4_cuda.o src/ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c src/ds4.h src/ds4_gpu_args.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o src/ds4_gpu_args.o src/ds4_kvstore.o vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c src/ds4.h src/ds4_gpu_args.h src/ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o src/ds4_cuda_test_hooks.o src/ds4_gpu_args.o src/ds4_kvstore.o vendor/rax.o src/ds4_distributed.o src/ds4_ssd.o src/ds4_cuda.o src/ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch

ds4_test: ds4_test.o src/ds4_help.o src/ds4_kvstore.o vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o src/ds4_help.o src/ds4_kvstore.o vendor/rax.o $(CORE_OBJS) $(CUDA_LDLIBS)

test: ds4_test ds4-eval q4k-dot-test mxfp4-dot-test \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	$(SAMPLING_TEST) ds4 ds4-server ds4-bench
	./ds4-eval --self-test-extractors
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	./tests/test_sampling

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./scripts/download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_cpu ds4_native ds4_server_test ds4_test tools/gguf-tools/quality-testing/score_official tools/gguf-tools/quality-testing/score_official.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_cuda tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o src/*.o src/cuda/mmq/*.o vendor/*.o tests/cuda_long_context_smoke
