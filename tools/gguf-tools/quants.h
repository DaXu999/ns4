#ifndef NS4_QUANTS_H
#define NS4_QUANTS_H

/*
 * Narrow quantization API used by the NS4 GGUF writer.
 *
 * The enum values intentionally match GGUF/GGML type IDs so template metadata
 * can be copied without translation.  Only the formats used by the NS4 Flash
 * quantization recipes are implemented as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define NS4Q_MAX_DIMS 4

typedef enum {
    NS4Q_TYPE_F32     = 0,
    NS4Q_TYPE_F16     = 1,
    NS4Q_TYPE_Q4_0    = 2,
    NS4Q_TYPE_Q4_1    = 3,
    NS4Q_TYPE_Q5_0    = 6,
    NS4Q_TYPE_Q5_1    = 7,
    NS4Q_TYPE_Q8_0    = 8,
    NS4Q_TYPE_Q8_1    = 9,
    NS4Q_TYPE_Q2_K    = 10,
    NS4Q_TYPE_Q3_K    = 11,
    NS4Q_TYPE_Q4_K    = 12,
    NS4Q_TYPE_Q5_K    = 13,
    NS4Q_TYPE_Q6_K    = 14,
    NS4Q_TYPE_Q8_K    = 15,
    NS4Q_TYPE_IQ2_XXS = 16,
    NS4Q_TYPE_IQ2_XS  = 17,
    NS4Q_TYPE_IQ3_XXS = 18,
    NS4Q_TYPE_IQ1_S   = 19,
    NS4Q_TYPE_IQ4_NL  = 20,
    NS4Q_TYPE_IQ3_S   = 21,
    NS4Q_TYPE_IQ2_S   = 22,
    NS4Q_TYPE_IQ4_XS  = 23,
    NS4Q_TYPE_I8      = 24,
    NS4Q_TYPE_I16     = 25,
    NS4Q_TYPE_I32     = 26,
    NS4Q_TYPE_I64     = 27,
    NS4Q_TYPE_F64     = 28,
    NS4Q_TYPE_IQ1_M   = 29,
    NS4Q_TYPE_BF16    = 30,
    NS4Q_TYPE_TQ1_0   = 34,
    NS4Q_TYPE_TQ2_0   = 35,
    NS4Q_TYPE_MXFP4   = 39,
    NS4Q_TYPE_NVFP4   = 40,
    NS4Q_TYPE_Q1_0    = 41,
    NS4Q_TYPE_COUNT   = 42,
} ns4q_type;

static inline size_t ns4q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *ns4q_type_name(ns4q_type type);
bool ns4q_can_quantize(ns4q_type type);
int64_t ns4q_block_size(ns4q_type type);
size_t ns4q_row_size(ns4q_type type, int64_t ne);
bool ns4q_requires_imatrix(ns4q_type type);
void ns4q_quantize_init(ns4q_type type);
size_t ns4q_quantize_chunk(ns4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

float ns4q_f16_to_f32(uint16_t bits);
float ns4q_bf16_to_f32(uint16_t bits);
void ns4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void ns4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

#endif
