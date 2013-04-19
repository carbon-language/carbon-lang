// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=178

// RUN: %clangxx_asan -m64 -O0 %p/SharedLibs/init-order-dlopen-so.cc \
// RUN:     -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -m64 -O0 %s -o %t -Wl,--export-dynamic
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1 | FileCheck %s
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
