// Check that unloading a module doesn't break coverage dumping for remaining
// modules.
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSHARED %s -shared -o %T/libcoverage_module_unloaded_test_1.so -fPIC
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSHARED %s -shared -o %T/libcoverage_module_unloaded_test_2.so -fPIC
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSO_DIR=\"%T\" %s -o %t
// RUN: export ASAN_OPTIONS=coverage=1:verbosity=1
// RUN: mkdir -p %T/coverage-module-unloaded && cd %T/coverage-module-unloaded
// RUN: %run %t 2>&1         | FileCheck %s
// RUN: %run %t foo 2>&1         | FileCheck %s
// RUN: cd .. && rm coverage-module-unloaded -r
//
// https://code.google.com/p/address-sanitizer/issues/detail?id=263
// XFAIL: android

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

#ifdef SHARED
extern "C" {
void bar() { printf("bar\n"); }
}
#else

int main(int argc, char **argv) {
  fprintf(stderr, "PID: %d\n", getpid());
  void *handle1 =
      dlopen(SO_DIR "/libcoverage_module_unloaded_test_1.so", RTLD_LAZY);
  assert(handle1);
  void (*bar1)() = (void (*)())dlsym(handle1, "bar");
  assert(bar1);
  bar1();
  void *handle2 =
      dlopen(SO_DIR "/libcoverage_module_unloaded_test_2.so", RTLD_LAZY);
  assert(handle2);
  void (*bar2)() = (void (*)())dlsym(handle2, "bar");
  assert(bar2);
  bar2();

  // It matters whether the unloaded module has a higher or lower address range
  // than the remaining one. Make sure to test both cases.
  if (argc < 2)
    dlclose(bar1 < bar2 ? handle1 : handle2);
  else
    dlclose(bar1 < bar2 ? handle2 : handle1);
  return 0;
}
#endif

// CHECK: PID: [[PID:[0-9]+]]
// CHECK: [[PID]].sancov: 1 PCs written
// CHECK: .so.[[PID]]
// If we get coverage for both DSOs, it means the module wasn't unloaded and
// this test is useless.
// CHECK-NOT: .so.[[PID]]
