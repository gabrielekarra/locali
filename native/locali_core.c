#include "locali_core.h"

#include <stdlib.h>
#include <string.h>

enum {
    LOCALI_NONE = 0,
    LOCALI_PROBATION = 1,
    LOCALI_PROTECTED = 2,
};

struct locali_cache {
    int32_t key_capacity;
    int32_t slots;
    int32_t probation_capacity;
    int32_t protected_capacity;
    int32_t probation_size;
    int32_t protected_size;
    int32_t probation_head;
    int32_t probation_tail;
    int32_t protected_head;
    int32_t protected_tail;
    int32_t free_count;
    int32_t policy;
    uint32_t decode_tokens;
    uint32_t pin_epoch;
    uint64_t clock;
    int32_t *slot_by_key;
    int32_t *key_by_slot;
    int32_t *prev;
    int32_t *next;
    int32_t *free_slots;
    uint8_t *segment;
    uint32_t *hotness;
    uint64_t *last_used;
    uint32_t *pin_marks;
};

static int valid_key(const locali_cache *cache, int32_t key) {
    return cache && key >= 0 && key < cache->key_capacity;
}

static void list_remove(locali_cache *cache, int32_t key) {
    const uint8_t segment = cache->segment[key];
    int32_t *head = segment == LOCALI_PROTECTED
        ? &cache->protected_head : &cache->probation_head;
    int32_t *tail = segment == LOCALI_PROTECTED
        ? &cache->protected_tail : &cache->probation_tail;
    int32_t *size = segment == LOCALI_PROTECTED
        ? &cache->protected_size : &cache->probation_size;
    const int32_t prev = cache->prev[key];
    const int32_t next = cache->next[key];
    if (prev >= 0) cache->next[prev] = next;
    else *head = next;
    if (next >= 0) cache->prev[next] = prev;
    else *tail = prev;
    cache->prev[key] = -1;
    cache->next[key] = -1;
    cache->segment[key] = LOCALI_NONE;
    (*size)--;
}

static void list_append(locali_cache *cache, int32_t key, uint8_t segment) {
    int32_t *head = segment == LOCALI_PROTECTED
        ? &cache->protected_head : &cache->probation_head;
    int32_t *tail = segment == LOCALI_PROTECTED
        ? &cache->protected_tail : &cache->probation_tail;
    int32_t *size = segment == LOCALI_PROTECTED
        ? &cache->protected_size : &cache->probation_size;
    cache->prev[key] = *tail;
    cache->next[key] = -1;
    if (*tail >= 0) cache->next[*tail] = key;
    else *head = key;
    *tail = key;
    cache->segment[key] = segment;
    (*size)++;
}

static void touch(locali_cache *cache, int32_t key) {
    const uint8_t segment = cache->segment[key];
    if (segment == LOCALI_PROTECTED) {
        list_remove(cache, key);
        list_append(cache, key, LOCALI_PROTECTED);
        return;
    }
    if (cache->protected_capacity == 0) {
        list_remove(cache, key);
        list_append(cache, key, LOCALI_PROBATION);
        return;
    }
    list_remove(cache, key);
    list_append(cache, key, LOCALI_PROTECTED);
    if (cache->protected_size > cache->protected_capacity) {
        const int32_t demote = cache->protected_head;
        list_remove(cache, demote);
        list_append(cache, demote, LOCALI_PROBATION);
    }
}

static int slot_is_pinned(const locali_cache *cache, int32_t slot) {
    return slot >= 0 && cache->pin_marks[slot] == cache->pin_epoch;
}

static int32_t first_unpinned(const locali_cache *cache,
                              int32_t head) {
    for (int32_t key = head; key >= 0; key = cache->next[key]) {
        if (!slot_is_pinned(cache, cache->slot_by_key[key])) return key;
    }
    return -1;
}

static int claim(locali_cache *cache,
                 int32_t key,
                 int32_t *slot,
                 int32_t *evicted_key) {
    const int probation_full =
        cache->probation_size >= cache->probation_capacity;
    int32_t victim = -1;
    if (probation_full || cache->free_count == 0) {
        victim = first_unpinned(cache, cache->probation_head);
    }

    if (victim < 0 && !probation_full && cache->free_count > 0) {
        *slot = cache->free_slots[--cache->free_count];
    } else {
        if (victim < 0) {
            victim = first_unpinned(cache, cache->protected_head);
        }
        if (victim < 0) return 0;
        *slot = cache->slot_by_key[victim];
        *evicted_key = victim;
        list_remove(cache, victim);
        cache->slot_by_key[victim] = -1;
        cache->key_by_slot[*slot] = -1;
    }

    cache->slot_by_key[key] = *slot;
    cache->key_by_slot[*slot] = key;
    list_append(cache, key, LOCALI_PROBATION);
    return 1;
}

static int32_t lfu_victim(const locali_cache *cache) {
    int32_t victim = -1;
    uint32_t lowest = UINT32_MAX;
    uint64_t oldest = UINT64_MAX;
    for (int32_t key = cache->probation_head;
         key >= 0;
         key = cache->next[key]) {
        const int32_t slot = cache->slot_by_key[key];
        if (slot_is_pinned(cache, slot)) continue;
        if (cache->hotness[key] < lowest ||
            (cache->hotness[key] == lowest &&
             cache->last_used[key] < oldest)) {
            victim = key;
            lowest = cache->hotness[key];
            oldest = cache->last_used[key];
        }
    }
    return victim;
}

static int claim_lfu(locali_cache *cache,
                     int32_t key,
                     int32_t *slot,
                     int32_t *evicted_key) {
    if (cache->free_count > 0) {
        *slot = cache->free_slots[--cache->free_count];
    } else {
        const int32_t victim = lfu_victim(cache);
        if (victim < 0) return 0;
        *slot = cache->slot_by_key[victim];
        *evicted_key = victim;
        list_remove(cache, victim);
        cache->slot_by_key[victim] = -1;
        cache->key_by_slot[*slot] = -1;
    }
    cache->slot_by_key[key] = *slot;
    cache->key_by_slot[*slot] = key;
    list_append(cache, key, LOCALI_PROBATION);
    return 1;
}

locali_cache *locali_cache_create(int32_t key_capacity,
                                  int32_t slots,
                                  int32_t protected_capacity) {
    return locali_cache_create_policy(
        key_capacity, slots, protected_capacity, LOCALI_CACHE_SLRU
    );
}

locali_cache *locali_cache_create_policy(int32_t key_capacity,
                                         int32_t slots,
                                         int32_t protected_capacity,
                                         int32_t policy) {
    if (key_capacity <= 0 || slots < 2 || protected_capacity < 0 ||
        protected_capacity >= slots ||
        (policy != LOCALI_CACHE_SLRU &&
         policy != LOCALI_CACHE_LFU_DECAY)) return NULL;
    if (policy == LOCALI_CACHE_LFU_DECAY) protected_capacity = 0;
    locali_cache *cache = calloc(1, sizeof(*cache));
    if (!cache) return NULL;
    cache->key_capacity = key_capacity;
    cache->slots = slots;
    cache->policy = policy;
    cache->protected_capacity = protected_capacity;
    cache->probation_capacity = slots - 1 - protected_capacity;
    cache->probation_head = cache->probation_tail = -1;
    cache->protected_head = cache->protected_tail = -1;
    cache->slot_by_key = malloc((size_t)key_capacity * sizeof(int32_t));
    cache->key_by_slot = malloc((size_t)slots * sizeof(int32_t));
    cache->prev = malloc((size_t)key_capacity * sizeof(int32_t));
    cache->next = malloc((size_t)key_capacity * sizeof(int32_t));
    cache->free_slots = malloc((size_t)(slots - 1) * sizeof(int32_t));
    cache->segment = calloc((size_t)key_capacity, sizeof(uint8_t));
    cache->hotness = calloc((size_t)key_capacity, sizeof(uint32_t));
    cache->last_used = calloc((size_t)key_capacity, sizeof(uint64_t));
    cache->pin_marks = calloc((size_t)slots, sizeof(uint32_t));
    if (!cache->slot_by_key || !cache->key_by_slot || !cache->prev ||
        !cache->next || !cache->free_slots || !cache->segment ||
        !cache->hotness || !cache->last_used || !cache->pin_marks) {
        locali_cache_destroy(cache);
        return NULL;
    }
    for (int32_t i = 0; i < key_capacity; i++) {
        cache->slot_by_key[i] = -1;
        cache->prev[i] = -1;
        cache->next[i] = -1;
    }
    for (int32_t i = 0; i < slots; i++) cache->key_by_slot[i] = -1;
    for (int32_t i = 1; i < slots; i++) cache->free_slots[i - 1] = i;
    cache->free_count = slots - 1;
    return cache;
}

void locali_cache_destroy(locali_cache *cache) {
    if (!cache) return;
    free(cache->slot_by_key);
    free(cache->key_by_slot);
    free(cache->prev);
    free(cache->next);
    free(cache->free_slots);
    free(cache->segment);
    free(cache->hotness);
    free(cache->last_used);
    free(cache->pin_marks);
    free(cache);
}

void locali_cache_note_token(locali_cache *cache) {
    if (!cache || cache->policy != LOCALI_CACHE_LFU_DECAY) return;
    cache->decode_tokens++;
    if (cache->decode_tokens % 16u != 0u) return;
    for (int32_t key = 0; key < cache->key_capacity; key++) {
        cache->hotness[key] >>= 1;
    }
}

int locali_cache_plan(locali_cache *cache,
                      const int32_t *keys,
                      size_t key_count,
                      const int32_t *pinned_slots,
                      size_t pinned_count,
                      int touch_hits,
                      locali_cache_result *results) {
    if (!cache || (!keys && key_count) || (!results && key_count)) return -1;
    cache->pin_epoch++;
    if (cache->pin_epoch == 0) {
        memset(cache->pin_marks, 0, (size_t)cache->slots * sizeof(uint32_t));
        cache->pin_epoch = 1;
    }
    for (size_t i = 0; i < pinned_count; i++) {
        const int32_t slot = pinned_slots[i];
        if (slot > 0 && slot < cache->slots) {
            cache->pin_marks[slot] = cache->pin_epoch;
        }
    }
    for (size_t i = 0; i < key_count; i++) {
        const int32_t key = keys[i];
        locali_cache_result result = {
            .key = key,
            .slot = -1,
            .evicted_key = -1,
            .hit = 0,
            .placed = 0,
        };
        if (!valid_key(cache, key)) {
            return -3;
        }
        if (cache->policy == LOCALI_CACHE_LFU_DECAY) {
            if (cache->hotness[key] < UINT32_MAX) cache->hotness[key]++;
            cache->last_used[key] = ++cache->clock;
        }
        if (cache->slot_by_key[key] >= 0) {
            result.hit = 1;
            result.placed = 1;
            result.slot = cache->slot_by_key[key];
            if (touch_hits && cache->policy == LOCALI_CACHE_SLRU) {
                touch(cache, key);
            }
        } else {
            result.placed = (uint8_t)(
                cache->policy == LOCALI_CACHE_LFU_DECAY
                ? claim_lfu(
                    cache, key, &result.slot, &result.evicted_key)
                : claim(
                    cache, key, &result.slot, &result.evicted_key)
            );
        }
        results[i] = result;
    }
    return 0;
}

int32_t locali_cache_peek(const locali_cache *cache, int32_t key) {
    return valid_key(cache, key) ? cache->slot_by_key[key] : -1;
}

int32_t locali_cache_probation_size(const locali_cache *cache) {
    return cache ? cache->probation_size : -1;
}

int32_t locali_cache_protected_size(const locali_cache *cache) {
    return cache ? cache->protected_size : -1;
}
