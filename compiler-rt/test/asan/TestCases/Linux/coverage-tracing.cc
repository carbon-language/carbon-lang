// Test -mllvm -sanitizer-coverage-experimental-tracing
//
// RUN: %clangxx_asan -O1 -fsanitize-coverage=2 -mllvm -sanitizer-coverage-experimental-tracing %s -o %t
// RUN: rm -rf   %T/coverage-tracing
// RUN: mkdir -p %T/coverage-tracing
// RUN: ASAN_OPTIONS=coverage=1:coverage_dir=%T/coverage-tracing:verbosity=1 %run %t 1 2 3 4 2>&1 | FileCheck %s
// RUN: rm -rf   %T/coverage-tracing
//
// REQUIRES: asan-64-bits

volatile int sink;
int main(int argc, char **argv) {
  volatile int i = 0;
  do {
    sink = 0;
    i++;
  } while (i < argc);
  return 0;
}

// CHECK: CovDump: Trace: {{[3-9]}} PCs written
// CHECK: CovDump: Trace: {{[6-9]}} Events written
