@ RUN: llvm-mc -triple armv8 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V8
@ RUN: llvm-mc -triple armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7
@ RUN: llvm-mc -triple armv6 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V6
setend be
@ CHECK-V8: warning: deprecated
@ CHECK-V7-NOT: warning: deprecated
mcr p15, #0, r5, c7, c5, #4
@ CHECK-V8: warning: deprecated since v7, use 'isb'
@ CHECK-V7: warning: deprecated since v7, use 'isb'
@ CHECK-V6-NOT: warning: deprecated since v7, use 'isb'
mcr p15, #0, r5, c7, c10, #4
@ CHECK-V8: warning: deprecated since v7, use 'dsb'
@ CHECK-V7: warning: deprecated since v7, use 'dsb'
@ CHECK-V6-NOT: warning: deprecated since v7, use 'dsb'
mcr p15, #0, r5, c7, c10, #5
@ CHECK-V8: warning: deprecated since v7, use 'dmb'
@ CHECK-V7: warning: deprecated since v7, use 'dmb'
@ CHECK-V6-NOT: warning: deprecated since v7, use 'dmb'
