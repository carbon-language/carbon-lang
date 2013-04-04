// Check that __asan_poison_memory_region works.
// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
//
// Check that we can disable it
// RUN: ASAN_OPTIONS=allow_user_poisoning=0 %t

#include <stdlib.h>

extern "C" void __asan_poison_memory_region(void *, size_t);

int main(int argc, char **argv) {
  char *x = new char[16];
  x[10] = 0;
  __asan_poison_memory_region(x, 16);
  int res = x[argc * 10];  // BOOOM
  // CHECK: ERROR: AddressSanitizer: use-after-poison
  // CHECK: main{{.*}}use-after-poison.cc:[[@LINE-2]]
  delete [] x;
  return res;
}
