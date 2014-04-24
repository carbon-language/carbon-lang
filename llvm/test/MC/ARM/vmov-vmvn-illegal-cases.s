@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text

@ CHECK: error: invalid operand for instruction
@ CHECK: vmov.i32        d2, #0xffffffab
@ CHECK: error: invalid operand for instruction
@ CHECK: vmov.i32        q2, #0xffffffab
@ CHECK: error: invalid operand for instruction
@ CHECK: vmov.i16        q2, #0xffab
@ CHECK: error: invalid operand for instruction
@ CHECK: vmov.i16        q2, #0xffab

@ CHECK: error: invalid operand for instruction
@ CHECK: vmvn.i32        d2, #0xffffffab
@ CHECK: error: invalid operand for instruction
@ CHECK: vmvn.i32        q2, #0xffffffab
@ CHECK: error: invalid operand for instruction
@ CHECK: vmvn.i16        q2, #0xffab
@ CHECK: error: invalid operand for instruction
@ CHECK: vmvn.i16        q2, #0xffab

        vmov.i32        d2, #0xffffffab
        vmov.i32        q2, #0xffffffab
        vmov.i16        q2, #0xffab
        vmov.i16        q2, #0xffab

        vmvn.i32        d2, #0xffffffab
        vmvn.i32        q2, #0xffffffab
        vmvn.i16        q2, #0xffab
        vmvn.i16        q2, #0xffab
