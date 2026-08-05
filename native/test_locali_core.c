#include "locali_core.h"

#include <assert.h>
#include <stdio.h>

static locali_cache_result access_one(locali_cache *cache,
                                       int32_t key,
                                       int touch) {
    locali_cache_result result;
    assert(locali_cache_plan(cache, &key, 1, NULL, 0, touch, &result) == 0);
    return result;
}

static void test_lru(void) {
    locali_cache *cache = locali_cache_create(16, 4, 0);
    assert(cache);
    assert(access_one(cache, 1, 1).slot == 3);
    assert(access_one(cache, 2, 1).slot == 2);
    assert(access_one(cache, 3, 1).slot == 1);
    assert(access_one(cache, 1, 1).hit);
    locali_cache_result result = access_one(cache, 4, 1);
    assert(!result.hit && result.evicted_key == 2 && result.slot == 2);
    assert(locali_cache_peek(cache, 1) == 3);
    assert(locali_cache_peek(cache, 2) == -1);
    locali_cache_destroy(cache);
}

static void test_slru_reservation_and_promotion(void) {
    locali_cache *cache = locali_cache_create(16, 5, 2);
    assert(cache);
    assert(access_one(cache, 0, 1).slot == 4);
    assert(access_one(cache, 1, 1).slot == 3);
    locali_cache_result result = access_one(cache, 2, 1);
    assert(result.evicted_key == 0 && result.slot == 4);
    assert(locali_cache_probation_size(cache) == 2);
    assert(access_one(cache, 1, 1).hit);
    assert(locali_cache_probation_size(cache) == 1);
    assert(locali_cache_protected_size(cache) == 1);
    assert(access_one(cache, 3, 1).slot == 2);
    assert(access_one(cache, 2, 1).hit);
    assert(locali_cache_protected_size(cache) == 2);
    locali_cache_destroy(cache);
}

static void test_pinned_slots_are_not_reused(void) {
    locali_cache *cache = locali_cache_create(8, 3, 0);
    assert(cache);
    const int32_t keys[] = {0, 1};
    locali_cache_result initial[2];
    assert(locali_cache_plan(cache, keys, 2, NULL, 0, 1, initial) == 0);
    const int32_t pinned[] = {initial[0].slot, initial[1].slot};
    const int32_t next = 2;
    locali_cache_result result;
    assert(locali_cache_plan(cache, &next, 1, pinned, 2, 1, &result) == 0);
    assert(!result.placed && result.slot == -1);
    locali_cache_destroy(cache);
}

static void test_lfu_decay(void) {
    locali_cache *cache = locali_cache_create_policy(
        16, 3, 0, LOCALI_CACHE_LFU_DECAY
    );
    assert(cache);
    assert(!access_one(cache, 1, 1).hit);
    assert(access_one(cache, 1, 1).hit);
    assert(!access_one(cache, 2, 1).hit);
    locali_cache_result result = access_one(cache, 3, 1);
    assert(result.evicted_key == 2);
    for (int i = 0; i < 16; i++) locali_cache_note_token(cache);
    result = access_one(cache, 4, 1);
    assert(result.evicted_key == 3);
    assert(locali_cache_peek(cache, 1) >= 0);
    locali_cache_destroy(cache);
}

int main(void) {
    test_lru();
    test_slru_reservation_and_promotion();
    test_pinned_slots_are_not_reused();
    test_lfu_decay();
    puts("locali_core: all tests passed");
    return 0;
}
