// Test various levels of coverage
//
// RUN: %clangxx_asan -O1 -fsanitize-coverage=1  %s -o %t
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %clangxx_asan -O1 -fsanitize-coverage=2  %s -o %t
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2
// RUN: %clangxx_asan -O1 -fsanitize-coverage=3  %s -o %t
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3
//
// REQUIRES: asan-64-bits

volatile int sink;
int main(int argc, char **argv) {
  if (argc == 0)
    sink = 0;
}

// CHECK1:  1 PCs written
// CHECK2:  2 PCs written
// CHECK3:  3 PCs written
