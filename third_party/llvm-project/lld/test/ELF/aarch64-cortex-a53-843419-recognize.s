// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o
// RUN: ld.lld -fix-cortex-a53-843419 -z separate-code -verbose %t.o -o %t2 2>&1 | FileCheck -check-prefix CHECK-PRINT %s
// RUN: llvm-objdump --triple=aarch64-linux-gnu -d %t2 | FileCheck %s --check-prefixes=CHECK,CHECK-FIX
// RUN: ld.lld %t.o -z separate-code -o %t3
// RUN: llvm-objdump --triple=aarch64-linux-gnu -d %t3 | FileCheck %s --check-prefixes=CHECK,CHECK-NOFIX
// RUN: ld.lld -fix-cortex-a53-843419 -r -z separate-code %t.o -o %t4
// RUN: llvm-objdump --triple=aarch64-linux-gnu -d %t4 | FileCheck %s --check-prefixes=CHECK-RELOCATABLE
// Test cases for Cortex-A53 Erratum 843419
// See ARM-EPM-048406 Cortex_A53_MPCore_Software_Developers_Errata_Notice.pdf
// for full erratum details.
// In Summary
// 1.)
// ADRP (0xff8 or 0xffc).
// 2.)
// - load or store single register or either integer or vector registers.
// - STP or STNP of either vector or vector registers.
// - Advanced SIMD ST1 store instruction.
// - Must not write Rn.
// 3.) optional instruction, can't be a branch, must not write Rn, may read Rn.
// 4.) A load or store instruction from the Load/Store register unsigned
// immediate class using Rn as the base register.

// Each section contains a sequence of instructions that should be recognized
// as erratum 843419. The test cases cover the major variations such as:
// - adrp starts at 0xfff8 or 0xfffc.
// - Variations in instruction class for instruction 2.
// - Optional instruction 3 present or not.
// - Load or store for instruction 4.

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 211FF8 in unpatched output.
// CHECK: <t3_ff8_ldr>:
// CHECK-NEXT:    211ff8:        60 02 00 f0     adrp    x0, 0x260000
// CHECK-NEXT:    211ffc:        21 00 40 f9     ldr             x1, [x1]
// CHECK-FIX:     212000:        03 c8 00 14     b       0x24400c
// CHECK-NOFIX:   212000:        00 00 40 f9     ldr             x0, [x0]
// CHECK-NEXT:    212004:        c0 03 5f d6     ret
// CHECK-RELOCATABLE: <t3_ff8_ldr>:
// CHECK-RELOCATABLE-NEXT:    ff8:        00 00 00 90     adrp    x0, 0x0
// CHECK-RELOCATABLE-NEXT:    ffc:        21 00 40 f9     ldr             x1, [x1]
// CHECK-RELOCATABLE-NEXT:   1000:        00 00 40 f9     ldr             x0, [x0]
// CHECK-RELOCATABLE-NEXT:   1004:        c0 03 5f d6     ret

        .section .text.01, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldr
        .type t3_ff8_ldr, %function
        .space 4096 - 8
t3_ff8_ldr:
        adrp x0, dat1
        ldr x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 213FF8 in unpatched output.
// CHECK: <t3_ff8_ldrsimd>:
// CHECK-NEXT:    213ff8:        60 02 00 b0     adrp    x0, 0x260000
// CHECK-NEXT:    213ffc:        21 00 40 bd     ldr             s1, [x1]
// CHECK-FIX:     214000:        05 c0 00 14     b       0x244014
// CHECK-NOFIX:   214000:        02 04 40 f9     ldr     x2, [x0, #8]
// CHECK-NEXT:    214004:        c0 03 5f d6     ret
        .section .text.02, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldrsimd
        .type t3_ff8_ldrsimd, %function
        .space 4096 - 8
t3_ff8_ldrsimd:
        adrp x0, dat2
        ldr s1, [x1, #0]
        ldr x2, [x0, :got_lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 215FFC in unpatched output.
// CHECK: <t3_ffc_ldrpost>:
// CHECK-NEXT:    215ffc:        40 02 00 f0     adrp    x0, 0x260000
// CHECK-NEXT:    216000:        21 84 40 bc     ldr     s1, [x1], #8
// CHECK-FIX:     216004:        06 b8 00 14     b       0x24401c
// CHECK-NOFIX:   216004:        03 08 40 f9     ldr     x3, [x0, #16]
// CHECK-NEXT:    216008:        c0 03 5f d6     ret
        .section .text.03, "ax", %progbits
        .balign 4096
        .globl t3_ffc_ldrpost
        .type t3_ffc_ldrpost, %function
        .space 4096 - 4
t3_ffc_ldrpost:
        adrp x0, dat3
        ldr s1, [x1], #8
        ldr x3, [x0, :got_lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 217FF8 in unpatched output.
// CHECK: <t3_ff8_strpre>:
// CHECK-NEXT:    217ff8:        40 02 00 b0     adrp    x0, 0x260000
// CHECK-NEXT:    217ffc:        21 8c 00 bc     str     s1, [x1, #8]!
// CHECK-FIX:     218000:        09 b0 00 14     b       0x244024
// CHECK-NOFIX:   218000:        02 0c 40 f9     ldr     x2, [x0, #24]
// CHECK-NEXT:    218004:        c0 03 5f d6     ret
        .section .text.04, "ax", %progbits
        .balign 4096
        .globl t3_ff8_strpre
        .type t3_ff8_strpre, %function
        .space 4096 - 8
t3_ff8_strpre:
        adrp x0, dat1
        str s1, [x1, #8]!
        ldr x2, [x0, :lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 219FFC in unpatched output.
// CHECK: <t3_ffc_str>:
// CHECK-NEXT:    219ffc:        3c 02 00 f0     adrp    x28, 0x260000
// CHECK-NEXT:    21a000:        42 00 00 f9     str             x2, [x2]
// CHECK-FIX:     21a004:        0a a8 00 14     b       0x24402c
// CHECK-NOFIX:   21a004:        9c 13 00 f9     str     x28, [x28, #32]
// CHECK-NEXT:    21a008:        c0 03 5f d6     ret
        .section .text.05, "ax", %progbits
        .balign 4096
        .globl t3_ffc_str
        .type t3_ffc_str, %function
        .space 4096 - 4
t3_ffc_str:
        adrp x28, dat2
        str x2, [x2, #0]
        str x28, [x28, :lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 21BFFC in unpatched output.
// CHECK: <t3_ffc_strsimd>:
// CHECK-NEXT:    21bffc:        3c 02 00 b0     adrp    x28, 0x260000
// CHECK-NEXT:    21c000:        44 00 00 b9     str             w4, [x2]
// CHECK-FIX:     21c004:        0c a0 00 14     b       0x244034
// CHECK-NOFIX:   21c004:        84 17 00 f9     str     x4, [x28, #40]
// CHECK-NEXT:    21c008:        c0 03 5f d6     ret
        .section .text.06, "ax", %progbits
        .balign 4096
        .globl t3_ffc_strsimd
        .type t3_ffc_strsimd, %function
        .space 4096 - 4
t3_ffc_strsimd:
        adrp x28, dat3
        str w4, [x2, #0]
        str x4, [x28, :lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 21DFF8 in unpatched output.
// CHECK: <t3_ff8_ldrunpriv>:
// CHECK-NEXT:    21dff8:        1d 02 00 f0     adrp    x29, 0x260000
// CHECK-NEXT:    21dffc:        41 08 40 38     ldtrb           w1, [x2]
// CHECK-FIX:     21e000:        0f 98 00 14     b       0x24403c
// CHECK-NOFIX:   21e000:        bd 03 40 f9     ldr             x29, [x29]
// CHECK-NEXT:    21e004:        c0 03 5f d6     ret
        .section .text.07, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldrunpriv
        .type t3_ff8_ldrunpriv, %function
        .space 4096 - 8
t3_ff8_ldrunpriv:
        adrp x29, dat1
        ldtrb w1, [x2, #0]
        ldr x29, [x29, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 21FFFC in unpatched output.
// CHECK: <t3_ffc_ldur>:
// CHECK-NEXT:    21fffc:        1d 02 00 b0     adrp    x29, 0x260000
// CHECK-NEXT:    220000:        42 40 40 b8     ldur    w2, [x2, #4]
// CHECK-FIX:     220004:        10 90 00 14     b       0x244044
// CHECK-NOFIX:   220004:        bd 07 40 f9     ldr     x29, [x29, #8]
// CHECK-NEXT:    220008:        c0 03 5f d6     ret
        .balign 4096
        .globl t3_ffc_ldur
        .type t3_ffc_ldur, %function
        .space 4096 - 4
t3_ffc_ldur:
        adrp x29, dat2
        ldur w2, [x2, #4]
        ldr x29, [x29, :got_lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 221FFC in unpatched output.
// CHECK: <t3_ffc_sturh>:
// CHECK-NEXT:    221ffc:        f2 01 00 f0     adrp    x18, 0x260000
// CHECK-NEXT:    222000:        43 40 00 78     sturh   w3, [x2, #4]
// CHECK-FIX:     222004:        12 88 00 14     b       0x24404c
// CHECK-NOFIX:   222004:        41 0a 40 f9     ldr     x1, [x18, #16]
// CHECK-NEXT:    222008:        c0 03 5f d6     ret
        .section .text.09, "ax", %progbits
        .balign 4096
        .globl t3_ffc_sturh
        .type t3_ffc_sturh, %function
        .space 4096 - 4
t3_ffc_sturh:
        adrp x18, dat3
        sturh w3, [x2, #4]
        ldr x1, [x18, :got_lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 223FF8 in unpatched output.
// CHECK: <t3_ff8_literal>:
// CHECK-NEXT:    223ff8:        f2 01 00 b0     adrp    x18, 0x260000
// CHECK-NEXT:    223ffc:        e3 ff ff 58     ldr     x3, 0x223ff8
// CHECK-FIX:     224000:        15 80 00 14     b       0x244054
// CHECK-NOFIX:   224000:        52 0e 40 f9     ldr     x18, [x18, #24]
// CHECK-NEXT:    224004:        c0 03 5f d6     ret
        .section .text.10, "ax", %progbits
        .balign 4096
        .globl t3_ff8_literal
        .type t3_ff8_literal, %function
        .space 4096 - 8
t3_ff8_literal:
        adrp x18, dat1
        ldr x3, t3_ff8_literal
        ldr x18, [x18, :lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 225FFC in unpatched output.
// CHECK: <t3_ffc_register>:
// CHECK-NEXT:    225ffc:        cf 01 00 f0     adrp    x15, 0x260000
// CHECK-NEXT:    226000:        43 68 61 f8     ldr             x3, [x2, x1]
// CHECK-FIX:     226004:        16 78 00 14     b       0x24405c
// CHECK-NOFIX:   226004:        ea 11 40 f9     ldr     x10, [x15, #32]
// CHECK-NEXT:    226008:        c0 03 5f d6     ret
        .section .text.11, "ax", %progbits
        .balign 4096
        .globl t3_ffc_register
        .type t3_ffc_register, %function
        .space 4096 - 4
t3_ffc_register:
        adrp x15, dat2
        ldr x3, [x2, x1]
        ldr x10, [x15, :lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 227FF8 in unpatched output.
// CHECK: <t3_ff8_stp>:
// CHECK-NEXT:    227ff8:        d0 01 00 b0     adrp    x16, 0x260000
// CHECK-NEXT:    227ffc:        61 08 00 a9     stp             x1, x2, [x3]
// CHECK-FIX:     228000:        19 70 00 14     b       0x244064
// CHECK-NOFIX:   228000:        0d 16 40 f9     ldr     x13, [x16, #40]
// CHECK-NEXT:    228004:        c0 03 5f d6     ret
        .section .text.12, "ax", %progbits
        .balign 4096
        .globl t3_ff8_stp
        .type t3_ff8_stp, %function
        .space 4096 - 8
t3_ff8_stp:
        adrp x16, dat3
        stp x1,x2, [x3, #0]
        ldr x13, [x16, :lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 229FFC in unpatched output.
// CHECK: <t3_ffc_stnp>:
// CHECK-NEXT:    229ffc:        a7 01 00 f0     adrp    x7, 0x260000
// CHECK-NEXT:    22a000:        61 08 00 a8     stnp            x1, x2, [x3]
// CHECK-FIX:     22a004:        1a 68 00 14     b       0x24406c
// CHECK-NOFIX:   22a004:        e9 0c 40 f9     ldr             x9, [x7, #24]
// CHECK-NEXT:    22a008:        c0 03 5f d6     ret
        .section .text.13, "ax", %progbits
        .balign 4096
        .globl t3_ffc_stnp
        .type t3_ffc_stnp, %function
        .space 4096 - 4
t3_ffc_stnp:
        adrp x7, dat1
        stnp x1,x2, [x3, #0]
        ldr x9, [x7, :lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 22BFFC in unpatched output.
// CHECK: <t3_ffc_st1singlepost>:
// CHECK-NEXT:    22bffc:        b7 01 00 b0     adrp    x23, 0x260000
// CHECK-NEXT:    22c000:        20 04 82 0d     st1 { v0.b }[1], [x1], x2
// CHECK-FIX:     22c004:        1c 60 00 14     b       0x244074
// CHECK-NOFIX:   22c004:        f6 12 40 f9     ldr     x22, [x23, #32]
// CHECK-NEXT:    22c008:        c0 03 5f d6     ret
        .section .text.14, "ax", %progbits
        .balign 4096
        .globl t3_ffc_st1singlepost
        .type t3_ffc_st1singlepost, %function
        .space 4096 - 4
t3_ffc_st1singlepost:
        adrp x23, dat2
        st1 { v0.b }[1], [x1], x2
        ldr x22, [x23, :lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 22DFF8 in unpatched output.
// CHECK: <t3_ff8_st1multiple>:
// CHECK-NEXT:    22dff8:        97 01 00 f0     adrp    x23, 0x260000
// CHECK-NEXT:    22dffc:        20 a0 00 4c     st1     { v0.16b, v1.16b }, [x1]
// CHECK-FIX:     22e000:        1f 58 00 14     b       0x24407c
// CHECK-NOFIX:   22e000:        f8 16 40 f9     ldr     x24, [x23, #40]
// CHECK-NEXT:    22e004:        c0 03 5f d6     ret
        .section .text.15, "ax", %progbits
        .balign 4096
        .globl t3_ff8_st1multiple
        .type t3_ff8_st1muliple, %function
        .space 4096 - 8
t3_ff8_st1multiple:
        adrp x23, dat3
        st1 { v0.16b, v1.16b }, [x1]
        ldr x24, [x23, :lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 22FFF8 in unpatched output.
// CHECK: <t4_ff8_ldr>:
// CHECK-NEXT:    22fff8:        80 01 00 b0     adrp    x0, 0x260000
// CHECK-NEXT:    22fffc:        21 00 40 f9     ldr             x1, [x1]
// CHECK-NEXT:    230000:        42 00 00 8b     add             x2, x2, x0
// CHECK-FIX:     230004:        20 50 00 14     b       0x244084
// CHECK-NOFIX:   230004:        02 00 40 f9     ldr             x2, [x0]
// CHECK-NEXT:    230008:        c0 03 5f d6     ret
        .section .text.16, "ax", %progbits
        .balign 4096
        .globl t4_ff8_ldr
        .type t4_ff8_ldr, %function
        .space 4096 - 8
t4_ff8_ldr:
        adrp x0, dat1
        ldr x1, [x1, #0]
        add x2, x2, x0
        ldr x2, [x0, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 231FFC in unpatched output.
// CHECK: <t4_ffc_str>:
// CHECK-NEXT:    231ffc:        7c 01 00 f0     adrp    x28, 0x260000
// CHECK-NEXT:    232000:        42 00 00 f9     str             x2, [x2]
// CHECK-NEXT:    232004:        20 00 02 cb     sub             x0, x1, x2
// CHECK-FIX:     232008:        21 48 00 14     b       0x24408c
// CHECK-NOFIX:   232008:        9b 07 00 f9     str     x27, [x28, #8]
// CHECK-NEXT:    23200c:        c0 03 5f d6     ret
        .section .text.17, "ax", %progbits
        .balign 4096
        .globl t4_ffc_str
        .type t4_ffc_str, %function
        .space 4096 - 4
t4_ffc_str:
        adrp x28, dat2
        str x2, [x2, #0]
        sub x0, x1, x2
        str x27, [x28, :got_lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 233FF8 in unpatched output.
// CHECK: <t4_ff8_stp>:
// CHECK-NEXT:    233ff8:        70 01 00 b0     adrp    x16, 0x260000
// CHECK-NEXT:    233ffc:        61 08 00 a9     stp             x1, x2, [x3]
// CHECK-NEXT:    234000:        03 7e 10 9b     mul             x3, x16, x16
// CHECK-FIX:     234004:        24 40 00 14     b       0x244094
// CHECK-NOFIX:   234004:        0e 0a 40 f9     ldr     x14, [x16, #16]
// CHECK-NEXT:    234008:        c0 03 5f d6     ret
        .section .text.18, "ax", %progbits
        .balign 4096
        .globl t4_ff8_stp
        .type t4_ff8_stp, %function
        .space 4096 - 8
t4_ff8_stp:
        adrp x16, dat3
        stp x1,x2, [x3, #0]
        mul x3, x16, x16
        ldr x14, [x16, :got_lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 235FF8 in unpatched output.
// CHECK: <t4_ff8_stppre>:
// CHECK-NEXT:    235ff8:        50 01 00 f0     adrp    x16, 0x260000
// CHECK-NEXT:    235ffc:        61 08 81 a9     stp     x1, x2, [x3, #16]!
// CHECK-NEXT:    236000:        03 7e 10 9b     mul             x3, x16, x16
// CHECK-FIX:     236004:        26 38 00 14     b       0x24409c
// CHECK-NOFIX:   236004:        0e 06 40 f9     ldr     x14, [x16, #8]
// CHECK-NEXT:    236008:        c0 03 5f d6     ret
        .section .text.19, "ax", %progbits
        .balign 4096
        .globl t4_ff8_stppre
        .type t4_ff8_stppre, %function
        .space 4096 - 8
t4_ff8_stppre:
        adrp x16, dat1
        stp x1,x2, [x3, #16]!
        mul x3, x16, x16
        ldr x14, [x16, #8]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 237FF8 in unpatched output.
// CHECK: <t4_ff8_stppost>:
// CHECK-NEXT:    237ff8:        50 01 00 b0     adrp    x16, 0x260000
// CHECK-NEXT:    237ffc:        61 08 81 a8     stp     x1, x2, [x3], #16
// CHECK-NEXT:    238000:        03 7e 10 9b     mul             x3, x16, x16
// CHECK-FIX:     238004:        28 30 00 14     b       0x2440a4
// CHECK-NOFIX:   238004:        0e 06 40 f9     ldr     x14, [x16, #8]
// CHECK-NEXT:    238008:        c0 03 5f d6     ret
        .section .text.20, "ax", %progbits
        .balign 4096
        .globl t4_ff8_stppost
        .type t4_ff8_stppost, %function
        .space 4096 - 8
t4_ff8_stppost:
        adrp x16, dat2
        stp x1,x2, [x3], #16
        mul x3, x16, x16
        ldr x14, [x16, #8]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 239FFC in unpatched output.
// CHECK: <t4_ffc_stpsimd>:
// CHECK-NEXT:    239ffc:        30 01 00 f0     adrp    x16, 0x260000
// CHECK-NEXT:    23a000:        61 08 00 ad     stp             q1, q2, [x3]
// CHECK-NEXT:    23a004:        03 7e 10 9b     mul             x3, x16, x16
// CHECK-FIX:     23a008:        29 28 00 14     b       0x2440ac
// CHECK-NOFIX:   23a008:        0e 06 40 f9     ldr     x14, [x16, #8]
// CHECK-NEXT:    23a00c:        c0 03 5f d6     ret
        .section .text.21, "ax", %progbits
        .balign 4096
        .globl t4_ffc_stpsimd
        .type t4_ffc_stpsimd, %function
        .space 4096 - 4
t4_ffc_stpsimd:
        adrp x16, dat3
        stp q1,q2, [x3, #0]
        mul x3, x16, x16
        ldr x14, [x16, #8]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 23BFFC in unpatched output.
// CHECK: <t4_ffc_stnp>:
// CHECK-NEXT:    23bffc:        27 01 00 b0     adrp    x7, 0x260000
// CHECK-NEXT:    23c000:        61 08 00 a8     stnp            x1, x2, [x3]
// CHECK-NEXT:    23c004:        1f 20 03 d5     nop
// CHECK-FIX:     23c008:        2b 20 00 14     b       0x2440b4
// CHECK-NOFIX:   23c008:        ea 00 40 f9     ldr             x10, [x7]
// CHECK-NEXT:    23c00c:        c0 03 5f d6     ret
        .section .text.22, "ax", %progbits
        .balign 4096
        .globl t4_ffc_stnp
        .type t4_ffc_stnp, %function
        .space 4096 - 4
t4_ffc_stnp:
        adrp x7, dat1
        stnp x1,x2, [x3, #0]
        nop
        ldr x10, [x7, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 23DFFC in unpatched output.
// CHECK: <t4_ffc_st1>:
// CHECK-NEXT:    23dffc:        18 01 00 f0     adrp    x24, 0x260000
// CHECK-NEXT:    23e000:        20 80 00 4d     st1 { v0.s }[2], [x1]
// CHECK-NEXT:    23e004:        f6 06 40 f9     ldr     x22, [x23, #8]
// CHECK-FIX:     23e008:        2d 18 00 14     b       0x2440bc
// CHECK-NOFIX:   23e008:        18 ff 3f f9     str     x24, [x24, #32760]
// CHECK-NEXT:    23e00c:        c0 03 5f d6     ret
        .section .text.23, "ax", %progbits
        .balign 4096
        .globl t4_ffc_st1
        .type t4_ffc_st1, %function
        .space 4096 - 4
t4_ffc_st1:
        adrp x24, dat2
        st1 { v0.s }[2], [x1]
        ldr x22, [x23, :got_lo12:dat2]
        str x24, [x24, #32760]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 23FFF8 in unpatched output.
// CHECK: <t3_ff8_ldr_once>:
// CHECK-NEXT:    23fff8:        00 01 00 b0     adrp    x0, 0x260000
// CHECK-NEXT:    23fffc:        20 70 82 4c     st1     { v0.16b }, [x1], x2
// CHECK-FIX:     240000:        31 10 00 14     b       0x2440c4
// CHECK-NOFIX:   240000:        01 08 40 f9     ldr     x1, [x0, #16]
// CHECK-NEXT:    240004:        02 08 40 f9     ldr     x2, [x0, #16]
// CHECK-NEXT:    240008:        c0 03 5f d6     ret
        .section .text.24, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldr_once
        .type t3_ff8_ldr_once, %function
        .space 4096 - 8
t3_ff8_ldr_once:
        adrp x0, dat3
        st1 { v0.16b }, [x1], x2
        ldr x1, [x0, #16]
        ldr x2, [x0, #16]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 241FF8 in unpatched output.
// CHECK: <t3_ff8_ldxr>:
// CHECK-NEXT:    241ff8:        e0 00 00 f0     adrp    x0, 0x260000
// CHECK-NEXT:    241ffc:        03 7c 5f c8     ldxr    x3, [x0]
// CHECK-FIX:     242000:        33 08 00 14     b       0x2440cc
// CHECK-NOFIX:   242000:        01 08 40 f9     ldr     x1, [x0, #16]
// CHECK:         242004:        02 08 40 f9     ldr     x2, [x0, #16]
// CHECK-NEXT:    242008:        c0 03 5f d6     ret
        .section .text.25, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldxr
        .type t3_ff8_ldxr, %function
        .space 4096 - 8
t3_ff8_ldxr:
        adrp x0, dat3
        ldxr x3, [x0]
        ldr x1, [x0, #16]
        ldr x2, [x0, #16]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 243FF8 in unpatched output.
// CHECK: <t3_ff8_stxr>:
// CHECK-NEXT:    243ff8:        e0 00 00 b0     adrp    x0, 0x260000
// CHECK-NEXT:    243ffc:        03 7c 04 c8     stxr    w4, x3, [x0]
// CHECK-FIX:     244000:        35 00 00 14     b       0x2440d4
// CHECK-NOFIX:   244000:        01 08 40 f9     ldr     x1, [x0, #16]
// CHECK:         244004:        02 08 40 f9     ldr     x2, [x0, #16]
// CHECK-NEXT:    244008:        c0 03 5f d6     ret
        .section .text.26, "ax", %progbits
        .balign 4096
        .globl t3_ff8_stxr
        .type t3_ff8_stxr, %function
        .space 4096 - 8
t3_ff8_stxr:
        adrp x0, dat3
        stxr w4, x3, [x0]
        ldr x1, [x0, #16]
        ldr x2, [x0, #16]
        ret

        .text
        .globl _start
        .type _start, %function
_start:
        ret

// CHECK-FIX: <__CortexA53843419_212000>:
// CHECK-FIX-NEXT:    24400c:    00 00 40 f9     ldr     x0, [x0]
// CHECK-FIX-NEXT:    244010:    fd 37 ff 17     b       0x212004
// CHECK-FIX: <__CortexA53843419_214000>:
// CHECK-FIX-NEXT:    244014:    02 04 40 f9     ldr     x2, [x0, #8]
// CHECK-FIX-NEXT:    244018:    fb 3f ff 17     b       0x214004
// CHECK-FIX: <__CortexA53843419_216004>:
// CHECK-FIX-NEXT:    24401c:    03 08 40 f9     ldr     x3, [x0, #16]
// CHECK-FIX-NEXT:    244020:    fa 47 ff 17     b       0x216008
// CHECK-FIX: <__CortexA53843419_218000>:
// CHECK-FIX-NEXT:    244024:    02 0c 40 f9     ldr     x2, [x0, #24]
// CHECK-FIX-NEXT:    244028:    f7 4f ff 17     b       0x218004
// CHECK-FIX: <__CortexA53843419_21A004>:
// CHECK-FIX-NEXT:    24402c:    9c 13 00 f9     str     x28, [x28, #32]
// CHECK-FIX-NEXT:    244030:    f6 57 ff 17     b       0x21a008
// CHECK-FIX: <__CortexA53843419_21C004>:
// CHECK-FIX-NEXT:    244034:    84 17 00 f9     str     x4, [x28, #40]
// CHECK-FIX-NEXT:    244038:    f4 5f ff 17     b       0x21c008
// CHECK-FIX: <__CortexA53843419_21E000>:
// CHECK-FIX-NEXT:    24403c:    bd 03 40 f9     ldr     x29, [x29]
// CHECK-FIX-NEXT:    244040:    f1 67 ff 17     b       0x21e004
// CHECK-FIX: <__CortexA53843419_220004>:
// CHECK-FIX-NEXT:    244044:    bd 07 40 f9     ldr     x29, [x29, #8]
// CHECK-FIX-NEXT:    244048:    f0 6f ff 17     b       0x220008
// CHECK-FIX: <__CortexA53843419_222004>:
// CHECK-FIX-NEXT:    24404c:    41 0a 40 f9     ldr     x1, [x18, #16]
// CHECK-FIX-NEXT:    244050:    ee 77 ff 17     b       0x222008
// CHECK-FIX: <__CortexA53843419_224000>:
// CHECK-FIX-NEXT:    244054:    52 0e 40 f9     ldr     x18, [x18, #24]
// CHECK-FIX-NEXT:    244058:    eb 7f ff 17     b       0x224004
// CHECK-FIX: <__CortexA53843419_226004>:
// CHECK-FIX-NEXT:    24405c:    ea 11 40 f9     ldr     x10, [x15, #32]
// CHECK-FIX-NEXT:    244060:    ea 87 ff 17     b       0x226008
// CHECK-FIX: <__CortexA53843419_228000>:
// CHECK-FIX-NEXT:    244064:    0d 16 40 f9     ldr     x13, [x16, #40]
// CHECK-FIX-NEXT:    244068:    e7 8f ff 17     b       0x228004
// CHECK-FIX: <__CortexA53843419_22A004>:
// CHECK-FIX-NEXT:    24406c:    e9 0c 40 f9     ldr     x9, [x7, #24]
// CHECK-FIX-NEXT:    244070:    e6 97 ff 17     b       0x22a008
// CHECK-FIX: <__CortexA53843419_22C004>:
// CHECK-FIX-NEXT:    244074:    f6 12 40 f9     ldr     x22, [x23, #32]
// CHECK-FIX-NEXT:    244078:    e4 9f ff 17     b       0x22c008
// CHECK-FIX: <__CortexA53843419_22E000>:
// CHECK-FIX-NEXT:    24407c:    f8 16 40 f9     ldr     x24, [x23, #40]
// CHECK-FIX-NEXT:    244080:    e1 a7 ff 17     b       0x22e004
// CHECK-FIX: <__CortexA53843419_230004>:
// CHECK-FIX-NEXT:    244084:    02 00 40 f9     ldr     x2, [x0]
// CHECK-FIX-NEXT:    244088:    e0 af ff 17     b       0x230008
// CHECK-FIX: <__CortexA53843419_232008>:
// CHECK-FIX-NEXT:    24408c:    9b 07 00 f9     str     x27, [x28, #8]
// CHECK-FIX-NEXT:    244090:    df b7 ff 17     b       0x23200c
// CHECK-FIX: <__CortexA53843419_234004>:
// CHECK-FIX-NEXT:    244094:    0e 0a 40 f9     ldr     x14, [x16, #16]
// CHECK-FIX-NEXT:    244098:    dc bf ff 17     b       0x234008
// CHECK-FIX: <__CortexA53843419_236004>:
// CHECK-FIX-NEXT:    24409c:    0e 06 40 f9     ldr     x14, [x16, #8]
// CHECK-FIX-NEXT:    2440a0:    da c7 ff 17     b       0x236008
// CHECK-FIX: <__CortexA53843419_238004>:
// CHECK-FIX-NEXT:    2440a4:    0e 06 40 f9     ldr     x14, [x16, #8]
// CHECK-FIX-NEXT:    2440a8:    d8 cf ff 17     b       0x238008
// CHECK-FIX: <__CortexA53843419_23A008>:
// CHECK-FIX-NEXT:    2440ac:    0e 06 40 f9     ldr     x14, [x16, #8]
// CHECK-FIX-NEXT:    2440b0:    d7 d7 ff 17     b       0x23a00c
// CHECK-FIX: <__CortexA53843419_23C008>:
// CHECK-FIX-NEXT:    2440b4:    ea 00 40 f9     ldr     x10, [x7]
// CHECK-FIX-NEXT:    2440b8:    d5 df ff 17     b       0x23c00c
// CHECK-FIX: <__CortexA53843419_23E008>:
// CHECK-FIX-NEXT:    2440bc:    18 ff 3f f9     str     x24, [x24, #32760]
// CHECK-FIX-NEXT:    2440c0:    d3 e7 ff 17     b       0x23e00c
// CHECK-FIX: <__CortexA53843419_240000>:
// CHECK-FIX-NEXT:    2440c4:    01 08 40 f9     ldr     x1, [x0, #16]
// CHECK-FIX-NEXT:    2440c8:    cf ef ff 17     b       0x240004
// CHECK-FIX: <__CortexA53843419_242000>:
// CHECK-FIX-NEXT:    2440cc:    01 08 40 f9     ldr     x1, [x0, #16]
// CHECK-FIX-NEXT:    2440d0:    cd f7 ff 17     b       0x242004
// CHECK-FIX: <__CortexA53843419_244000>:
// CHECK-FIX-NEXT:    2440d4:    01 08 40 f9     ldr     x1, [x0, #16]
// CHECK-FIX-NEXT:    2440d8:    cb ff ff 17     b       0x244004
        .data
        .globl dat
        .globl dat2
        .globl dat3
dat1:   .quad 1
dat2:   .quad 2
dat3:   .quad 3
