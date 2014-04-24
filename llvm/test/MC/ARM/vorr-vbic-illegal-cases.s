@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text

@ CHECK: error: invalid operand for instruction
@ CHECK: vorr.i32        d2, #0xffffffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vorr.i32        q2, #0xffffffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vorr.i32        d2, #0xabababab
@ CHECK: error: invalid operand for instruction
@ CHECK: vorr.i32        q2, #0xabababab
@ CHECK: error: invalid operand for instruction
@ CHECK: vorr.i16        q2, #0xabab
@ CHECK: error: invalid operand for instruction
@ CHECK: vorr.i16        q2, #0xabab

@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        d2, #0xffffffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        q2, #0xffffffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        d2, #0xabababab
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        q2, #0xabababab
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i16        d2, #0xabab
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i16        q2, #0xabab

        vorr.i32        d2, #0xffffffff
        vorr.i32        q2, #0xffffffff
        vorr.i32        d2, #0xabababab
        vorr.i32        q2, #0xabababab
        vorr.i16        q2, #0xabab
        vorr.i16        q2, #0xabab

        vbic.i32        d2, #0xffffffff
        vbic.i32        q2, #0xffffffff
        vbic.i32        d2, #0xabababab
        vbic.i32        q2, #0xabababab
        vbic.i16        d2, #0xabab
        vbic.i16        q2, #0xabab
