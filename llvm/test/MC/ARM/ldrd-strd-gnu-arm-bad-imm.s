@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text
@ CHECK: error: instruction requires: thumb2
@ CHECK:         ldrd    r0, [r0, #512]
        ldrd    r0, [r0, #512]

@ CHECK: error: instruction requires: thumb2
@ CHECK:         strd    r0, [r0, #512]
        strd    r0, [r0, #512]
