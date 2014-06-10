// Test for direct coverage writing with dlopen.
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSHARED %s -shared -o %T/libcoverage_direct_test_1.so -fPIC
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSO_DIR=\"%device\" %s -o %t

// RUN: adb shell rm -rf %device/coverage-direct
// RUN: rm -rf %T/coverage-direct

// RUN: adb shell mkdir -p %device/coverage-direct/direct
// RUN: mkdir -p %T/coverage-direct/direct
// RUN: ASAN_OPTIONS=coverage=1:coverage_direct=1:coverage_dir=%device/coverage-direct/direct:verbosity=1 %run %t
// RUN: adb pull %device/coverage-direct/direct %T/coverage-direct/direct
// RUN: ls; pwd
// RUN: cd %T/coverage-direct/direct
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov

// FIXME: FileCheck disabled due to flakiness in the test. Fix and re-enable.
// ... |& FileCheck %s

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
      dlopen(SO_DIR "/libcoverage_direct_test_1.so", RTLD_LAZY);
  assert(handle1);
  void (*bar1)() = (void (*)())dlsym(handle1, "bar");
  assert(bar1);
  bar1();

  return 0;
}
#endif

// CHECK: 2 PCs total
