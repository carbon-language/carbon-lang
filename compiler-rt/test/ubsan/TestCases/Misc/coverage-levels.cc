// Test various levels of coverage
//
// RUN: mkdir -p %T/coverage-levels
// RUN: OPT=coverage=1:verbosity=1:coverage_dir=%T/coverage-levels
// RUN: %clangxx -fsanitize=shift                        -DGOOD_SHIFT=1 -O1 -fsanitize-coverage=1  %s -o %t
// RUN: UBSAN_OPTIONS=$OPT ASAN_OPTIONS=$OPT %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_NOWARN
// RUN: %clangxx -fsanitize=undefined                    -DGOOD_SHIFT=1 -O1 -fsanitize-coverage=1  %s -o %t
// RUN: UBSAN_OPTIONS=$OPT ASAN_OPTIONS=$OPT %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_NOWARN

// RUN: %clangxx -fsanitize=shift -O1 -fsanitize-coverage=1  %s -o %t
// RUN: UBSAN_OPTIONS=$OPT ASAN_OPTIONS=$OPT %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_WARN
// RUN: %clangxx -fsanitize=shift -O1 -fsanitize-coverage=2  %s -o %t
// RUN: UBSAN_OPTIONS=$OPT ASAN_OPTIONS=$OPT %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2 --check-prefix=CHECK_WARN
// RUN: %clangxx -fsanitize=shift -O1 -fsanitize-coverage=3  %s -o %t
// RUN: UBSAN_OPTIONS=$OPT ASAN_OPTIONS=$OPT %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3 --check-prefix=CHECK_WARN

// XFAIL: darwin

volatile int sink;
int main(int argc, char **argv) {
  int shift = argc * 32;
#if GOOD_SHIFT
  shift = 3;
#endif
  if ((argc << shift) == 16)  // False.
    return 1;
  return 0;
}

// CHECK_WARN: shift exponent 32 is too large
// CHECK_NOWARN-NOT: ERROR
// FIXME: Currently, coverage instrumentation kicks in after ubsan, so we get
// more than the minimal number of instrumented blocks.
// FIXME: Currently, ubsan with -fno-sanitize-recover and w/o asan will fail
// to dump coverage.
// CHECK1:  1 PCs written
// CHECK2:  3 PCs written
// CHECK3:  4 PCs written
