#ifndef NS4_SSD_H
#define NS4_SSD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} ns4_ssd_memory_lock;

bool ns4_parse_gib_arg(const char *s, uint64_t *bytes);
bool ns4_ssd_memory_lock_acquire(ns4_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void ns4_ssd_memory_lock_release(ns4_ssd_memory_lock *lock);

#endif
