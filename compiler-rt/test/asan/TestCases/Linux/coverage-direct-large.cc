// Test for direct coverage writing with lots of data.
// Current implementation maps output file in chunks of 64K. This test overflows
// 1 chunk.
// RUN: %clangxx_asan -mllvm -asan-coverage=1 -O0 %s -o %t

// RUN: rm -rf %T/coverage-direct-large

// RUN: mkdir -p %T/coverage-direct-large/normal && cd %T/coverage-direct-large/normal
// RUN: ASAN_OPTIONS=coverage=1:coverage_direct=0:verbosity=1 %run %t
// RUN: %sancov print *.sancov >out.txt
// RUN: cd ../..

// RUN: mkdir -p %T/coverage-direct-large/direct && cd %T/coverage-direct-large/direct
// RUN: ASAN_OPTIONS=coverage=1:coverage_direct=1:verbosity=1 %run %t
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

#define DECL(x) __attribute__((noinline)) void x() {}
#define CALL(x) x();

F4(DECL, f)

int main(void) {
  F4(CALL, f)
  return 0;
}
