// Make sure coverage is dumped even if there are reported leaks.
//
// RUN: %clangxx_asan -mllvm -asan-coverage=1 %s -o %t
//
// RUN: rm -rf %T/coverage-and-lsan
//
// RUN: mkdir -p %T/coverage-and-lsan/normal
// RUN: ASAN_OPTIONS=coverage=1:coverage_dir=%T/coverage-and-lsan:verbosity=1 not %run %t 2>&1 | FileCheck %s
// RUN: %sancov print %T/coverage-and-lsan/*.sancov 2>&1
//
// REQUIRES: asan-64-bits

int *g = new int;
int main(int argc, char **argv) {
  g = 0;
  return 0;
}

// CHECK: LeakSanitizer: detected memory leaks
// CHECK: CovDump:
