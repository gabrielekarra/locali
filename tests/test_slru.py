from collections import OrderedDict

from m25_arena import ArenaStore


def make_store():
    store = ArenaStore.__new__(ArenaStore)
    store.cache_policy = "slru-all"
    store.slots = {"cold": 5}  # four usable: 2 probation + 2 protected
    store.free = {"cold": [1, 2, 3, 4]}
    store.lru = {"cold": OrderedDict()}
    store.protected = {"cold": OrderedDict()}
    store.protected_cap = {"cold": 2}
    store.pinned = {}
    store.inflight = {}
    store.pf_keys = set()
    store.evictions = 0
    store.prefetch_wasted = 0
    return store


def test_slru_reserves_capacity_until_an_entry_earns_protection():
    store = make_store()
    a, b, c = (0, 1), (0, 2), (0, 3)

    store._claim("cold", a)
    store._claim("cold", b)
    assert len(store.free["cold"]) == 2

    # The probation window is full. A third one-hit entry replaces its oldest
    # member instead of consuming the half reserved for repeat requests.
    store._claim("cold", c)
    assert a not in store.lru["cold"]
    assert list(store.lru["cold"]) == [b, c]
    assert len(store.free["cold"]) == 2


def test_slru_promotes_hits_and_demotes_the_oldest_protected_entry():
    store = make_store()
    keys = [(0, i) for i in range(5)]
    store._claim("cold", keys[0])
    store._claim("cold", keys[1])
    store._touch("cold", keys[0])
    store._claim("cold", keys[2])
    store._touch("cold", keys[1])

    assert list(store.protected["cold"]) == keys[:2]

    store._touch("cold", keys[2])
    assert list(store.protected["cold"]) == keys[1:3]
    assert keys[0] in store.lru["cold"]

    # A new scan entry evicts probation before either repeated entry.
    store._claim("cold", keys[3])
    assert keys[1] in store.protected["cold"]
    assert keys[2] in store.protected["cold"]
