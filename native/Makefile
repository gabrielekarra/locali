CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LIB := liblocali_core.dylib
SHARED_FLAGS := -dynamiclib
else
LIB := liblocali_core.so
SHARED_FLAGS := -shared -fPIC
endif

.PHONY: all test clean

all: $(LIB)

$(LIB): locali_core.c locali_core.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) locali_core.c -o $@

test_locali_core: test_locali_core.c locali_core.c locali_core.h
	$(CC) $(CFLAGS) test_locali_core.c locali_core.c -o $@

test: test_locali_core
	./test_locali_core

clean:
	rm -f $(LIB) test_locali_core
