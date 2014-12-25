// Test for direct coverage writing enabled as activation time.

// RUN: %clangxx_asan -fsanitize-coverage=1 -DSHARED %s -shared -o %T/libcoverage_direct_test_1.so -fPIC
// RUN: %clangxx -c -DSO_DIR=\"%T\" %s -o %t.o
// RUN: %clangxx_asan -fsanitize-coverage=1 %t.o %libdl -o %t

// RUN: rm -rf %T/coverage-direct-activation

// RUN: mkdir -p %T/coverage-direct-activation/normal
// RUN: ASAN_OPTIONS=coverage=1,coverage_direct=0,coverage_dir=%T/coverage-direct-activation/normal:verbosity=1 %run %t
// RUN: %sancov print %T/coverage-direct-activation/normal/*.sancov >%T/coverage-direct-activation/normal/out.txt

// RUN: mkdir -p %T/coverage-direct-activation/direct
// RUN: ASAN_OPTIONS=start_deactivated=1,coverage_direct=1,verbosity=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=coverage=1,coverage_dir=%T/coverage-direct-activation/direct %run %t
// RUN: cd %T/coverage-direct-activation/direct
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: diff -u coverage-direct-activation/normal/out.txt coverage-direct-activation/direct/out.txt

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
