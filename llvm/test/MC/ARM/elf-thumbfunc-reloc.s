@@ test st_value bit 0 of thumb function
@ RUN: llvm-mc %s -triple=armv4t-freebsd-eabi -filetype=obj -o - | \
@ RUN: llvm-readobj -r  | FileCheck %s


	.syntax unified
        .text
        .globl  f
        .align  2
        .type   f,%function
        .code   16
        .thumb_func
f:
        push    {r7, lr}
        mov     r7, sp
        bl      g
        pop     {r7, pc}

@@ make sure an R_ARM_THM_CALL relocation is generated for the call to g
@CHECK:      Relocations [
@CHECK-NEXT:   Section (2) .rel.text {
@CHECK-NEXT:     0x4 R_ARM_THM_CALL g 0x0
@CHECK-NEXT:   }
@CHECK-NEXT: ]
