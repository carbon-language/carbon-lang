// Test for direct coverage writing with lots of data.
// Current implementation maps output file in chunks of 64K. This test overflows
// 1 chunk.

// RUN: %clangxx_asan -fsanitize-coverage=func -O0 -DSHARED %s -shared -o %dynamiclib -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func -O0 %s %libdl -o %t

// RUN: rm -rf %T/coverage-direct-large

// RUN: mkdir -p %T/coverage-direct-large/normal && cd %T/coverage-direct-large/normal
// RUN: %env_asan_opts=coverage=1:coverage_direct=0:verbosity=1 %run %t %dynamiclib
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: mkdir -p %T/coverage-direct-large/direct && cd %T/coverage-direct-large/direct
// RUN: %env_asan_opts=coverage=1:coverage_direct=1:verbosity=1 %run %t %dynamiclib
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: diff -u coverage-direct-large/normal/out.txt coverage-direct-large/direct/out.txt
//
// XFAIL: android

#define F0(Q, x) Q(x)
#define F1(Q, x)                                                          \
  F0(Q, x##0) F0(Q, x##1) F0(Q, x##2) F0(Q, x##3) F0(Q, x##4) F0(Q, x##5) \
      F0(Q, x##6) F0(Q, x##7) F0(Q, x##8) F0(Q, x##9)
#define F2(Q, x)                                                          \
  F1(Q, x##0) F1(Q, x##1) F1(Q, x##2) F1(Q, x##3) F1(Q, x##4) F1(Q, x##5) \
      F1(Q, x##6) F1(Q, x##7) F1(Q, x##8) F1(Q, x##9)
#define F3(Q, x)                                                          \
  F2(Q, x##0) F2(Q, x##1) F2(Q, x##2) F2(Q, x##3) F2(Q, x##4) F2(Q, x##5) \
      F2(Q, x##6) F2(Q, x##7) F2(Q, x##8) F2(Q, x##9)
#define F4(Q, x)                                                          \
  F3(Q, x##0) F3(Q, x##1) F3(Q, x##2) F3(Q, x##3) F3(Q, x##4) F3(Q, x##5) \
      F3(Q, x##6) F3(Q, x##7) F3(Q, x##8) F3(Q, x##9)

#define DECL(x) __attribute__((noinline)) static void x() {}
#define CALL(x) x();

F4(DECL, f)

#ifdef SHARED
extern "C" void so_entry() {
  F4(CALL, f)
}  
#else

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
  F4(CALL, f)
  assert(argc > 1);
  void *handle1 = dlopen(argv[1], RTLD_LAZY);  // %dynamiclib
  assert(handle1);
  void (*so_entry)() = (void (*)())dlsym(handle1, "so_entry");
  assert(so_entry);
  so_entry();

  return 0;
}

#endif // SHARED
