// Test that dynamically allocated TLS space is included in the root set.
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx %p/SharedLibs/huge_tls_lib_so.cc -fPIC -shared -o %t-so.so
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_tls=0" %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_tls=1" %t 2>&1
// RUN: LSAN_OPTIONS="" %t 2>&1

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

int main(int argc, char *argv[]) {
  std::string path = std::string(argv[0]) + "-so.so";

  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  assert(handle != 0);
  typedef void **(* store_t)(void *p);
  store_t StoreToTLS = (store_t)dlsym(handle, "StoreToTLS");
  assert(dlerror() == 0);

  void *p = malloc(1337);
  void **p_in_tls = StoreToTLS(p);
  assert(*p_in_tls == p);
  fprintf(stderr, "Test alloc: %p.\n", p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: leaked 1337 byte object at [[ADDR]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: SUMMARY: LeakSanitizer:
