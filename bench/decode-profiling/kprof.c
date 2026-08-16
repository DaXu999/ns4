/* LD_PRELOAD kernel profiler: wraps cudaLaunchKernel/cuLaunchKernel with a
 * pooled event pair per launch and aggregates GPU time per kernel symbol.
 *
 * Env:
 *   KPROF_SKIP=N    ignore the first N launches (prefill/warmup)
 *   KPROF_MAX=N     record at most N launches (default 120000)
 *   KPROF_OUT=path  where to write the report (default /tmp/kprof.txt)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

typedef cudaError_t (*launch_fn)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
typedef CUresult (*culaunch_fn)(CUfunction, unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned, unsigned,
                                CUstream, void **, void **);

static launch_fn real_launch;
static culaunch_fn real_culaunch;

typedef struct { const void *key; int driver; unsigned gx, gy, bx; cudaEvent_t a, b; } rec_t;

static rec_t *recs;
static long n_rec, max_rec = 120000, skip = 0, seen = 0;
static cudaEvent_t *pool;      /* 2 * max_rec preallocated events */
static long pool_next;
static int armed, dumped, in_dump;
static long dumps_done;   /* completed fill-dumps; see kprof_fini */

static double t0_ms, after_ms;

/* Host-side pipeline drains.  A cudaDeviceSynchronize or a blocking memcpy in
 * the per-token decode path costs far more than any kernel, because it empties
 * the queue and then refills it from scratch -- so it is worth counting them
 * separately from launches. */
static long n_devsync, n_strmsync, n_memcpy;
static double ms_devsync, ms_strmsync, ms_memcpy;
typedef cudaError_t (*devsync_fn)(void);
typedef cudaError_t (*strmsync_fn)(cudaStream_t);
typedef cudaError_t (*memcpy_fn)(void *, const void *, size_t, enum cudaMemcpyKind);
static devsync_fn real_devsync;
static strmsync_fn real_strmsync;
static memcpy_fn real_memcpy;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void kprof_init(void) {
    static int done;
    if (done) return;
    done = 1;
    const char *s;
    if ((s = getenv("KPROF_SKIP"))) skip = atol(s);
    if ((s = getenv("KPROF_MAX")))  max_rec = atol(s);
    if ((s = getenv("KPROF_AFTER_MS"))) after_ms = atof(s);
    t0_ms = now_ms();
    recs = (rec_t *)calloc((size_t)max_rec, sizeof(rec_t));
    pool = (cudaEvent_t *)calloc((size_t)max_rec * 2, sizeof(cudaEvent_t));
    if (!recs || !pool) { max_rec = 0; return; }
    for (long i = 0; i < max_rec * 2; i++) {
        if (cudaEventCreateWithFlags(&pool[i], cudaEventDefault) != cudaSuccess) {
            max_rec = i / 2;
            break;
        }
    }
    armed = 1;
    fprintf(stderr, "kprof: armed, pool=%ld pairs, skip=%ld\n", max_rec, skip);
}

static const char *sym_name(const void *key, int driver, char *buf, size_t n) {
    const char *nm = NULL;
    /* CUDA 13 hands __cudaLaunchKernel a cudaKernel_t handle, not a host
     * function pointer, so try the kernel-handle lookups first. */
    if (cuKernelGetName(&nm, (CUkernel)key) != CUDA_SUCCESS) nm = NULL;
    if (!nm && cuFuncGetName(&nm, (CUfunction)key) != CUDA_SUCCESS) nm = NULL;
    if (!nm && !driver && cudaFuncGetName(&nm, key) != cudaSuccess) nm = NULL;
    if (!nm) {
        Dl_info info;
        if (dladdr(key, &info) && info.dli_sname) nm = info.dli_sname;
    }
    if (!nm) { snprintf(buf, n, "%s%p", driver ? "drv:" : "rt:", key); return buf; }
    snprintf(buf, n, "%s", nm);
    return buf;
}

typedef struct { const void *key; int driver; unsigned gx, gy, bx; double ms, gap; long count; } agg_t;

static void kprof_dump(void) {
    if (dumped || !armed) return;
    dumped = 1; in_dump = 1;
    cudaDeviceSynchronize();
    const char *out = getenv("KPROF_OUT");
    FILE *f = fopen(out ? out : "/tmp/kprof.txt", "w");
    if (!f) { in_dump = 0; return; }

    agg_t *agg = (agg_t *)calloc((size_t)n_rec + 1, sizeof(agg_t));
    long n_agg = 0, n_fail = 0;
    double total = 0.0, total_gap = 0.0;
    cudaError_t first_err = cudaSuccess;
    for (long i = 0; i < n_rec; i++) {
        float ms = 0.0f;
        cudaError_t ee = cudaEventElapsedTime(&ms, recs[i].a, recs[i].b);
        if (ee != cudaSuccess) {
            if (first_err == cudaSuccess) first_err = ee;
            n_fail++;
            continue;
        }
        long j = 0;
        for (; j < n_agg; j++)
            if (agg[j].key == recs[i].key && agg[j].gx == recs[i].gx &&
                agg[j].gy == recs[i].gy && agg[j].bx == recs[i].bx) break;
        if (j == n_agg) {
            agg[n_agg].key = recs[i].key; agg[n_agg].driver = recs[i].driver;
            agg[n_agg].gx = recs[i].gx; agg[n_agg].gy = recs[i].gy;
            agg[n_agg].bx = recs[i].bx;
            n_agg++;
        }
        agg[j].ms += ms;
        agg[j].count++;
        total += ms;
        /* Idle gap in front of this launch: from the moment the GPU finished
         * the previous recorded kernel to the moment it started this one.  A
         * large gap means the GPU ran dry waiting for the host to submit --
         * a CPU-side stall, not a slow kernel. */
        if (i > 0) {
            float gp = 0.0f;
            if (cudaEventElapsedTime(&gp, recs[i-1].b, recs[i].a) == cudaSuccess) {
                agg[j].gap += gp;
                total_gap += gp;
            }
        }
    }
    for (long i = 0; i < n_agg; i++)
        for (long j = i + 1; j < n_agg; j++)
            if (agg[j].ms + agg[j].gap > agg[i].ms + agg[i].gap) { agg_t t = agg[i]; agg[i] = agg[j]; agg[j] = t; }

    fprintf(f, "kprof: %ld launches recorded, %.3f ms in kernels + "
               "%.3f ms idle gaps (%ld unreadable%s%s)\n",
            n_rec, total, total_gap, n_fail,
            n_fail ? ", first error: " : "",
            n_fail ? cudaGetErrorString(first_err) : "");
    fprintf(f, "%-58s %16s %9s %12s %10s %12s %10s\n", "kernel", "grid/block",
            "launches", "total_ms", "us/call", "gap_ms", "gap_us/call");
    char buf[512];
    for (long i = 0; i < n_agg; i++) {
        char shape[32];
        snprintf(shape, sizeof shape, "%ux%u/%u", agg[i].gx, agg[i].gy, agg[i].bx);
        fprintf(f, "%-58s %16s %9ld %12.3f %10.2f %12.3f %10.2f\n",
                sym_name(agg[i].key, agg[i].driver, buf, sizeof buf), shape,
                agg[i].count, agg[i].ms, agg[i].ms * 1000.0 / (double)agg[i].count,
                agg[i].gap, agg[i].gap * 1000.0 / (double)agg[i].count);
    }
    fprintf(f, "\nhost sync: cudaDeviceSynchronize %ld (%.3f ms), "
               "cudaStreamSynchronize %ld (%.3f ms), cudaMemcpy %ld (%.3f ms)\n",
            n_devsync, ms_devsync, n_strmsync, ms_strmsync, n_memcpy, ms_memcpy);
    fclose(f);
    dumps_done++;
    fprintf(stderr, "kprof: wrote %s (%ld launches, %.1f ms)\n", out ? out : "/tmp/kprof.txt", n_rec, total);
    in_dump = 0;
}

/* Only useful when the pool never filled; once KPROF_ROLL has written a real
 * window, the destructor must not clobber it with an unreadable one (every
 * cudaEventElapsedTime here fails with "driver shutting down"). */
__attribute__((destructor)) static void kprof_fini(void) {
    if (dumps_done) return;
    kprof_dump();
}

/* Recording an event on a stream that is mid-capture makes the capture
 * invalid ("invalid argument" at the next launch), so ns4 loses every decode
 * island and silently runs eager -- i.e. profiling would destroy the thing
 * being profiled.  Skip those launches: they are captured once and then
 * replayed via cudaGraphLaunch, which this interposer does not see anyway.
 * What survives in the report is exactly the set of launches NO graph
 * covers, which is the interesting set. */
static inline int stream_is_capturing(cudaStream_t s) {
    enum cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &st) != cudaSuccess) {
        (void)cudaGetLastError();
        return 1;   /* unsure -> stay out of the way */
    }
    return st != cudaStreamCaptureStatusNone;
}

static cudaError_t launch_common(launch_fn real, const void *func, dim3 g, dim3 b,
                                 void **args, size_t sh, cudaStream_t stream) {
    kprof_init();
    if (in_dump || !armed || ++seen <= skip || n_rec >= max_rec ||
        (after_ms > 0.0 && now_ms() - t0_ms < after_ms) ||
        stream_is_capturing(stream))
        return real(func, g, b, args, sh, stream);
    cudaEvent_t ea = pool[pool_next], eb = pool[pool_next + 1];
    pool_next += 2;
    cudaEventRecord(ea, stream);
    cudaError_t r = real(func, g, b, args, sh, stream);
    cudaEventRecord(eb, stream);
    recs[n_rec].key = func; recs[n_rec].driver = 0;
    recs[n_rec].gx = g.x; recs[n_rec].gy = g.y; recs[n_rec].bx = b.x;
    recs[n_rec].a = ea; recs[n_rec].b = eb;
    n_rec++;
    /* Dump as soon as the pool is full: the destructor runs after the CUDA
     * runtime has torn the context down, and every event read fails there.
     * KPROF_ROLL then recycles the pool so the file that survives describes
     * the LAST window of the run -- which is decode, after prefill is done. */
    if (n_rec >= max_rec) {
        kprof_dump();
        if (getenv("KPROF_ROLL")) {
            n_rec = 0; pool_next = 0; dumped = 0;
            /* per-window host-sync counts, so the surviving report
             * describes decode rather than prefill + decode */
            n_devsync = n_strmsync = n_memcpy = 0;
            ms_devsync = ms_strmsync = ms_memcpy = 0.0;
        }
    }
    return r;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 g, dim3 b, void **args,
                             size_t sh, cudaStream_t stream) {
    if (!real_launch) real_launch = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    return launch_common(real_launch, func, g, b, args, sh, stream);
}

/* CUDA 13's <<<>>> lowering calls this instead; the first argument is a
 * cudaKernel_t handle but it is pointer-sized and works as an aggregation key. */
static launch_fn real_launch2, real_launch2_ptsz;

cudaError_t __cudaLaunchKernel(const void *kernel, dim3 g, dim3 b, void **args,
                               size_t sh, cudaStream_t stream) {
    if (!real_launch2) real_launch2 = (launch_fn)dlsym(RTLD_NEXT, "__cudaLaunchKernel");
    return launch_common(real_launch2, kernel, g, b, args, sh, stream);
}

cudaError_t __cudaLaunchKernel_ptsz(const void *kernel, dim3 g, dim3 b, void **args,
                                    size_t sh, cudaStream_t stream) {
    if (!real_launch2_ptsz)
        real_launch2_ptsz = (launch_fn)dlsym(RTLD_NEXT, "__cudaLaunchKernel_ptsz");
    return launch_common(real_launch2_ptsz, kernel, g, b, args, sh, stream);
}

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned bz, unsigned sh,
                        CUstream stream, void **args, void **extra) {
    if (!real_culaunch) real_culaunch = (culaunch_fn)dlsym(RTLD_NEXT, "cuLaunchKernel");
    kprof_init();
    if (in_dump || !armed || ++seen <= skip || n_rec >= max_rec ||
        stream_is_capturing((cudaStream_t)stream))
        return real_culaunch(f, gx, gy, gz, bx, by, bz, sh, stream, args, extra);
    cudaEvent_t ea = pool[pool_next], eb = pool[pool_next + 1];
    pool_next += 2;
    cudaEventRecord(ea, (cudaStream_t)stream);
    CUresult r = real_culaunch(f, gx, gy, gz, bx, by, bz, sh, stream, args, extra);
    cudaEventRecord(eb, (cudaStream_t)stream);
    recs[n_rec].key = (const void *)f; recs[n_rec].driver = 1;
    recs[n_rec].gx = gx; recs[n_rec].gy = gy; recs[n_rec].bx = bx;
    recs[n_rec].a = ea; recs[n_rec].b = eb;
    n_rec++;
    return r;
}

cudaError_t cudaDeviceSynchronize(void) {
    if (!real_devsync) real_devsync = (devsync_fn)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    if (in_dump || !armed) return real_devsync();
    const double t = now_ms();
    cudaError_t r = real_devsync();
    ms_devsync += now_ms() - t;
    n_devsync++;
    return r;
}

cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    if (!real_strmsync) real_strmsync = (strmsync_fn)dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    if (in_dump || !armed) return real_strmsync(s);
    const double t = now_ms();
    cudaError_t r = real_strmsync(s);
    ms_strmsync += now_ms() - t;
    n_strmsync++;
    return r;
}

cudaError_t cudaMemcpy(void *d, const void *s, size_t n, enum cudaMemcpyKind k) {
    if (!real_memcpy) real_memcpy = (memcpy_fn)dlsym(RTLD_NEXT, "cudaMemcpy");
    if (in_dump || !armed) return real_memcpy(d, s, n, k);
    const double t = now_ms();
    cudaError_t r = real_memcpy(d, s, n, k);
    ms_memcpy += now_ms() - t;
    n_memcpy++;
    return r;
}
