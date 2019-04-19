// RUN: %clangxx_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 \
// RUN:   -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -O0 %s -o %t && \
// RUN:   not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 \
// RUN:   -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -O1 %s -o %t && \
// RUN:   not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 \
// RUN:   -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -O2 %s -o %t && \
// RUN:   not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 \
// RUN:   -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -O3 %s -o %t && \
// RUN:   not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// REQUIRES: aarch64-target-arch
#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

__attribute__((noinline)) void f(int *p) { *p = 3; }

// CHECK: ERROR: HWAddressSanitizer:
// CHECK: #0 {{.*}} in f(int*) {{.*}}register-dump-no-fp.cc:[[@LINE-3]]

int main() {
  __hwasan_enable_allocator_tagging();

  // Must come first - libc++ can clobber as it's not compiled with -ffixed-x10.
  int * volatile a = new int;

  asm volatile("mov x10, #0x2222\n"
               "mov x20, #0x3333\n"
               "mov x27, #0x4444\n");

  a = (int *)__hwasan_tag_pointer(a, 0);
  f(a);
  // CHECK: #1 {{.*}} in main {{.*}}register-dump-no-fp.cc:[[@LINE-1]]
  // CHECK: #2 {{.*}} in {{.*lib.*}}
}

// Developer note: FileCheck really doesn't like when you have a regex that
// ends with a '}' character, e.g. the regex "[0-9]{10}" will fail, because
// the closing '}' fails as an "unbalanced regex". We work around this by
// encasing the trailing space after a register, or the end-of-line specifier.
// CHECK: Registers where the failure occurred
// CHECK-NEXT: x0{{[ ]+[0-9a-f]{16}[ ]}}x1{{[ ]+[0-9a-f]{16}[ ]}}x2{{[ ]+[0-9a-f]{16}[ ]}}x3{{[ ]+[0-9a-f]{16}$}}
// CHECK-NEXT: x4{{[ ]+[0-9a-f]{16}[ ]}}x5{{[ ]+[0-9a-f]{16}[ ]}}x6{{[ ]+[0-9a-f]{16}[ ]}}x7{{[ ]+[0-9a-f]{16}$}}
// CHECK-NEXT: x8{{[ ]+[0-9a-f]{16}[ ]}}x9{{[ ]+[0-9a-f]{16}[ ]}}
// CHECK-SAME: x10 0000000000002222
// CHECK-SAME: x11{{[ ]+[0-9a-f]{16}$}}
// CHECK-NEXT: x12{{[ ]+[0-9a-f]{16}[ ]}}x13{{[ ]+[0-9a-f]{16}[ ]}}x14{{[ ]+[0-9a-f]{16}[ ]}}x15{{[ ]+[0-9a-f]{16}$}}
// CHECK-NEXT: x16{{[ ]+[0-9a-f]{16}[ ]}}x17{{[ ]+[0-9a-f]{16}[ ]}}x18{{[ ]+[0-9a-f]{16}[ ]}}x19{{[ ]+[0-9a-f]{16}$}}
// CHECK-NEXT: x20 0000000000003333
// CHECK-SAME: x21{{[ ]+[0-9a-f]{16}[ ]}}x22{{[ ]+[0-9a-f]{16}[ ]}}x23{{[ ]+[0-9a-f]{16}$}}
// CHECK-NEXT: x24{{[ ]+[0-9a-f]{16}[ ]}}x25{{[ ]+[0-9a-f]{16}[ ]}}x26{{[ ]+[0-9a-f]{16}[ ]}}
// CHECK-SAME: x27 0000000000004444
// CHECK-NEXT: x28{{[ ]+[0-9a-f]{16}[ ]}}x29{{[ ]+[0-9a-f]{16}[ ]}}x30{{[ ]+[0-9a-f]{16}$}}
