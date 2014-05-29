// Test for direct coverage writing with dlopen.
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSHARED %s -shared -o %T/libcoverage_direct_test_1.so -fPIC
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSO_DIR=\"%T\" %s -o %t

// RUN: rm -rf %T/coverage-direct

// RUN: mkdir -p %T/coverage-direct/normal
// RUN: ASAN_OPTIONS=coverage=1:coverage_direct=0:coverage_dir=%T/coverage-direct/normal:verbosity=1 %run %t
// RUN: %sancov print %T/coverage-direct/normal/*.sancov >%T/coverage-direct/normal/out.txt

// RUN: mkdir -p %T/coverage-direct/direct
// RUN: ASAN_OPTIONS=coverage=1:coverage_direct=1:coverage_dir=%T/coverage-direct/direct:verbosity=1 %run %t
// RUN: cd %T/coverage-direct/direct
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: diff -u coverage-direct/normal/out.txt coverage-direct/direct/out.txt
//
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
      dlopen(SO_DIR "/libcoverage_direct_test_1.so", RTLD_LAZY);
  assert(handle1);
  void (*bar1)() = (void (*)())dlsym(handle1, "bar");
  assert(bar1);
  bar1();

  return 0;
}
#endif
