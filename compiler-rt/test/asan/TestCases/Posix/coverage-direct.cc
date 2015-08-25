// Test for direct coverage writing with dlopen at coverage level 1 to 3.

// RUN: %clangxx_asan -fsanitize-coverage=func -DSHARED %s -shared -o %dynamiclib -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func %s %libdl -o %t

// RUN: rm -rf %T/coverage-direct

// RUN: mkdir -p %T/coverage-direct/normal
// RUN: %env_asan_opts=coverage=1:coverage_direct=0:coverage_dir=%T/coverage-direct/normal:verbosity=1 %run %t %dynamiclib
// RUN: %sancov print %T/coverage-direct/normal/*.sancov >%T/coverage-direct/normal/out.txt

// RUN: mkdir -p %T/coverage-direct/direct
// RUN: %env_asan_opts=coverage=1:coverage_direct=1:coverage_dir=%T/coverage-direct/direct:verbosity=1 %run %t %dynamiclib
// RUN: cd %T/coverage-direct/direct
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: diff -u coverage-direct/normal/out.txt coverage-direct/direct/out.txt


// RUN: %clangxx_asan -fsanitize-coverage=bb -DSHARED %s -shared -o %dynamiclib -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=bb -DSO_DIR=\"%T\" %s %libdl -o %t

// RUN: rm -rf %T/coverage-direct

// RUN: mkdir -p %T/coverage-direct/normal
// RUN: %env_asan_opts=coverage=1:coverage_direct=0:coverage_dir=%T/coverage-direct/normal:verbosity=1 %run %t %dynamiclib
// RUN: %sancov print %T/coverage-direct/normal/*.sancov >%T/coverage-direct/normal/out.txt

// RUN: mkdir -p %T/coverage-direct/direct
// RUN: %env_asan_opts=coverage=1:coverage_direct=1:coverage_dir=%T/coverage-direct/direct:verbosity=1 %run %t %dynamiclib
// RUN: cd %T/coverage-direct/direct
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: diff -u coverage-direct/normal/out.txt coverage-direct/direct/out.txt


// RUN: %clangxx_asan -fsanitize-coverage=edge -DSHARED %s -shared -o %dynamiclib -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=edge -DSO_DIR=\"%T\" %s %libdl -o %t

// RUN: rm -rf %T/coverage-direct

// RUN: mkdir -p %T/coverage-direct/normal
// RUN: %env_asan_opts=coverage=1:coverage_direct=0:coverage_dir=%T/coverage-direct/normal:verbosity=1 %run %t %dynamiclib
// RUN: %sancov print %T/coverage-direct/normal/*.sancov >%T/coverage-direct/normal/out.txt

// RUN: mkdir -p %T/coverage-direct/direct
// RUN: %env_asan_opts=coverage=1:coverage_direct=1:coverage_dir=%T/coverage-direct/direct:verbosity=1 %run %t %dynamiclib
// RUN: cd %T/coverage-direct/direct
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: diff -u coverage-direct/normal/out.txt coverage-direct/direct/out.txt

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
  assert(argc > 1);
  void *handle1 = dlopen(argv[1], RTLD_LAZY);
  assert(handle1);
  void (*bar1)() = (void (*)())dlsym(handle1, "bar");
  assert(bar1);
  bar1();

  return 0;
}
#endif
