// RUN: %clangxx_asan -O %s -o %t && %t
//
// Test __sanitizer_annotate_contiguous_container.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

extern "C" {
void __sanitizer_annotate_contiguous_container(void *beg, void *end,
                                               void *old_mid, void *new_mid);
bool __asan_address_is_poisoned(void *addr);
}  // extern "C"

void TestContainer(size_t capacity) {
  char *beg = new char[capacity];
  char *end = beg + capacity;
  char *mid = beg + capacity;
  char *old_mid = 0;

  for (int i = 0; i < 10000; i++) {
    size_t size = rand() % (capacity + 1);
    assert(size <= capacity);
    old_mid = mid;
    mid = beg + size;
    __sanitizer_annotate_contiguous_container(beg, end, old_mid, mid);

    for (size_t idx = 0; idx < size; idx++)
        assert(!__asan_address_is_poisoned(beg + idx));
    for (size_t idx = size; idx < capacity; idx++)
        assert(__asan_address_is_poisoned(beg + idx));
  }

  // Don't forget to unpoison the whole thing before destroing/reallocating.
  __sanitizer_annotate_contiguous_container(beg, end, mid, end);
  for (size_t idx = 0; idx < capacity; idx++)
    assert(!__asan_address_is_poisoned(beg + idx));
  delete[] beg;
}

__attribute__((noinline))
void Throw() { throw 1; }

__attribute__((noinline))
void ThrowAndCatch() {
  try {
    Throw();
  } catch(...) {
  }
}

void TestThrow() {
  char x[32];
  __sanitizer_annotate_contiguous_container(x, x + 32, x + 32, x + 14);
  assert(!__asan_address_is_poisoned(x + 13));
  assert(__asan_address_is_poisoned(x + 14));
  ThrowAndCatch();
  assert(!__asan_address_is_poisoned(x + 13));
  // FIXME: invert the assertion below once we fix
  // https://code.google.com/p/address-sanitizer/issues/detail?id=258
  assert(!__asan_address_is_poisoned(x + 14));
  __sanitizer_annotate_contiguous_container(x, x + 32, x + 14, x + 32);
  assert(!__asan_address_is_poisoned(x + 13));
  assert(!__asan_address_is_poisoned(x + 14));
}

int main(int argc, char **argv) {
  int n = argc == 1 ? 128 : atoi(argv[1]);
  for (int i = 0; i <= n; i++)
    TestContainer(i);
  TestThrow();
}
