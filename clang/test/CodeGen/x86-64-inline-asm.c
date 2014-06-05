// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64 %s -S -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple x86_64 %s -S -o /dev/null -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-Werror
void f() {
  asm("movaps %xmm3, (%esi, 2)"); // CHECK: warning: scale factor without index register is ignored
                                  // CHECK-Werror: error: scale factor without index register is ignored
}
