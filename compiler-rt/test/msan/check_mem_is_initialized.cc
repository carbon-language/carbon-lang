// RUN: %clangxx_msan -m64 -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O1 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O1 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out

#include <sanitizer/msan_interface.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int *volatile p = (int *)malloc(sizeof(int));

  __msan_check_mem_is_initialized(p, sizeof(*p));
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*check_mem_is_initialized.cc:}}[[@LINE-2]]

  // CHECK-ORIGINS: Uninitialized value was created by a heap allocation
  // CHECK-ORIGINS: {{#0 0x.* in .*malloc}}
  // CHECK-ORIGINS: {{#1 0x.* in main .*check_mem_is_initialized.cc:}}[[@LINE-8]]
  return 0;
}
