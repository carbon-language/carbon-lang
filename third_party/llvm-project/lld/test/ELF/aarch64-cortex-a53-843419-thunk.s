// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:          .text1 0x10000 : { *(.text.01) *(.text.02) *(.text.03) } \
// RUN:          .text2 0x8010000 : { *(.text.04) } } " > %t.script
// RUN: ld.lld --script %t.script -fix-cortex-a53-843419 -verbose %t.o -o %t2 \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-PRINT %s
// RUN: llvm-objdump --no-show-raw-insn --triple=aarch64-linux-gnu -d %t2 | FileCheck %s

/// %t2 is 128 Megabytes, so delete it early.
// RUN: rm %t2

/// Test cases for Cortex-A53 Erratum 843419 that involve interactions with
/// range extension thunks. Both erratum fixes and range extension thunks need
/// precise address information and after creation alter address information.


        .section .text.01, "ax", %progbits
        .balign 4096
        .globl _start
        .type _start, %function
_start:
        bl far_away
        /// Thunk to far_away, size 16-bytes goes here.

        .section .text.02, "ax", %progbits
        .space 4096 - 28

        /// Erratum sequence will only line up at address 0 modulo 0xffc when
        /// Thunk is inserted.
        .section .text.03, "ax", %progbits
        .globl t3_ff8_ldr
        .type t3_ff8_ldr, %function
t3_ff8_ldr:
        adrp x0, dat
        ldr x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 10FF8 in unpatched output.
// CHECK: 0000000000010ff8 <t3_ff8_ldr>:
// CHECK-NEXT: adrp    x0, 0x8010000
// CHECK-NEXT: ldr     x1, [x1]
// CHECK-NEXT: b       0x11008
// CHECK-NEXT: ret
// CHECK: 0000000000011008 <__CortexA53843419_11000>:
// CHECK-NEXT: ldr     x0, [x0, #8]
// CHECK-NEXT: b       0x11004
        .section .text.04, "ax", %progbits
        .globl far_away
        .type far_away, function
far_away:
        ret

        .section .data
        .globl dat
dat:    .quad 0
