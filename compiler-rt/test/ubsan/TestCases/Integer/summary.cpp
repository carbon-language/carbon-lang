// RUN: %clangxx -fsanitize=integer %s -o %t && %t 2>&1 | FileCheck %s
// REQUIRES: ubsan-asan

#include <stdint.h>

int main() {
  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
  // CHECK: SUMMARY: AddressSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]
  return 0;
}
