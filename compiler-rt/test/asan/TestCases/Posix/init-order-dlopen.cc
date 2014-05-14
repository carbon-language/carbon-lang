// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=178

// Assume we're on Darwin and try to pass -U to the linker. If this flag is
// unsupported, don't use it.
// RUN: %clangxx_asan -O0 -DSHARED_LIB %s \
// RUN:     -fPIC -shared -o %t-so.so -Wl,-U,_inc_global || \
// RUN:     %clangxx_asan -O0 -DSHARED_LIB %s \
// RUN:         -fPIC -shared -o %t-so.so
// If the linker doesn't support --export-dynamic (which is ELF-specific),
// try to link without that option.
// FIXME: find a better solution.
// RUN: %clangxx_asan -O0 %s -lpthread -ldl -o %t -Wl,--export-dynamic || \
// RUN:     %clangxx_asan -O0 %s -lpthread -ldl -o %t
// RUN: ASAN_OPTIONS=strict_init_order=true %run %t 2>&1 | FileCheck %s
#if !defined(SHARED_LIB)
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <string>

using std::string;

int foo() {
  return 42;
}
int global = foo();

__attribute__((visibility("default")))
extern "C"
void inc_global() {
  global++;
}

void *global_poller(void *arg) {
  while (true) {
    if (global != 42)
      break;
    usleep(100);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  pthread_t p;
  pthread_create(&p, 0, global_poller, 0);
  string path = string(argv[0]) + "-so.so";
  if (0 == dlopen(path.c_str(), RTLD_NOW)) {
    fprintf(stderr, "dlerror: %s\n", dlerror());
    return 1;
  }
  pthread_join(p, 0);
  printf("PASSED\n");
  // CHECK: PASSED
  return 0;
}
#else  // SHARED_LIB
#include <stdio.h>
#include <unistd.h>

extern "C" void inc_global();

int slow_init() {
  sleep(1);
  inc_global();
  return 42;
}

int slowly_init_glob = slow_init();
#endif  // SHARED_LIB
