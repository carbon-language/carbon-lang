// RUN: %clangxx -fsanitize=alignment %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOTYPE
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE
// REQUIRES: !ubsan-standalone && !ubsan-standalone-static

int main(int argc, char* argv[]) {
  __builtin_assume_aligned(argv, 0x80000000);
  // CHECK-NOTYPE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:28
  // CHECK-TYPE: SUMMARY: UndefinedBehaviorSanitizer: alignment-assumption {{.*}}summary.cpp:[[@LINE-2]]:28
  return 0;
}
