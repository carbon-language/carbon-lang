// RUN: %clangxx_asan -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

#include <string.h>

char kString[] = "foo";

int main(int argc, char **argv) {
  char *copy = strdup(kString);
  int x = copy[4 + argc];  // BOOM
  // CHECK: AddressSanitizer: heap-buffer-overflow
  // CHECK: #0 {{.*}}main {{.*}}strdup_oob_test.cc:[[@LINE-2]]
  // CHECK: allocated by thread T{{.*}} here:
  // CHECK: #0 {{.*}}strdup
  // CHECK: strdup_oob_test.cc:[[@LINE-6]]
  return x;
}
