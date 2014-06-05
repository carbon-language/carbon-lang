// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv8 -target-feature +neon %s -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-v8
// RUN: not %clang_cc1 -triple armv8 -target-feature +neon %s -S -o /dev/null -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-v8-Werror

void set_endian() {
  asm("setend be"); // CHECK-v8: warning: deprecated 
                    // CHECK-v8-Werror: error: deprecated
}
