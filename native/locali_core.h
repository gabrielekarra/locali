#ifndef LOCALI_CORE_H
#define LOCALI_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct locali_cache locali_cache;

enum {
    LOCALI_CACHE_SLRU = 0,
    LOCALI_CACHE_LFU_DECAY = 1,
};

typedef struct {
    int32_t key;
    int32_t slot;
    int32_t evicted_key;
    uint8_t hit;
    uint8_t placed;
} locali_cache_result;

locali_cache *locali_cache_create(int32_t key_capacity,
                                  int32_t slots,
                                  int32_t protected_capacity);
locali_cache *locali_cache_create_policy(int32_t key_capacity,
                                         int32_t slots,
                                         int32_t protected_capacity,
                                         int32_t policy);
void locali_cache_destroy(locali_cache *cache);
void locali_cache_note_token(locali_cache *cache);

int locali_cache_plan(locali_cache *cache,
                      const int32_t *keys,
                      size_t key_count,
                      const int32_t *pinned_slots,
                      size_t pinned_count,
                      int touch_hits,
                      locali_cache_result *results);

int32_t locali_cache_peek(const locali_cache *cache, int32_t key);
int32_t locali_cache_probation_size(const locali_cache *cache);
int32_t locali_cache_protected_size(const locali_cache *cache);

#ifdef __cplusplus
}
#endif

#endif
