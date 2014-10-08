// Test for ASAN_OPTIONS=start_deactivated=1 mode.
// Main executable is uninstrumented, but linked to ASan runtime. The shared
// library is instrumented. Memory errors before dlopen are not detected.

// RUN: %clangxx_asan -O0 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx -O0 %s -c -o %t.o
// RUN: %clangxx_asan -O0 %t.o -o %t
// RUN: ASAN_OPTIONS=start_deactivated=1 not %run %t 2>&1 | FileCheck %s
// XFAIL: arm-linux-gnueabi
// XFAIL: armv7l-unknown-linux-gnueabihf

#if !defined(SHARED_LIB)
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>

#include "sanitizer/asan_interface.h"

void test_malloc_shadow() {
  char *p = (char *)malloc(100);
  char *q = (char *)__asan_region_is_poisoned(p + 95, 8);
  fprintf(stderr, "=%zd=\n", q ? q - (p + 95) : -1);
  free(p);
}

typedef void (*Fn)();

int main(int argc, char *argv[]) {
  test_malloc_shadow();
  // CHECK: =-1=

  std::string path = std::string(argv[0]) + "-so.so";
  void *dso = dlopen(path.c_str(), RTLD_NOW);
  if (!dso) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  test_malloc_shadow();
  // CHECK: =5=

  void *fn = dlsym(dso, "do_another_bad_thing");
  if (!fn) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 1;
  }

  ((Fn)fn)();
  // CHECK: AddressSanitizer: heap-buffer-overflow
  // CHECK: READ of size 1
  // CHECK: {{#0 .* in do_another_bad_thing}}
  // CHECK: is located 5 bytes to the right of 100-byte region
  // CHECK: in do_another_bad_thing

  return 0;
}
#else  // SHARED_LIB
#include <stdio.h>
#include <stdlib.h>

extern "C" void do_another_bad_thing() {
  char *volatile p = (char *)malloc(100);
  printf("%hhx\n", p[105]);
}
#endif  // SHARED_LIB
