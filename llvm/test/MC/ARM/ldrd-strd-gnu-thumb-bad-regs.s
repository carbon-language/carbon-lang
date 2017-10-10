@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s

@ FIXME: These errors are inaccurate because the error is being reported on the
@ implicit r13 operand added after r12.

.text
.thumb
@ CHECK: error: operand must be a register in range [r0, r12] or r14
@ CHECK:         ldrd    r12, [r0, #512]
        ldrd    r12, [r0, #512]

@ CHECK: error: operand must be a register in range [r0, r12] or r14
@ CHECK:         strd    r12, [r0, #512]
        strd    r12, [r0, #512]
