# Instructions that are supposed to be invalid but currently aren't
# This test will XPASS if any insn stops assembling.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips4 \
# RUN:     2> %t1
# RUN: not FileCheck %s < %t1
# XFAIL: *

# CHECK-NOT: error
        .set noat
        deret
        di      $s8
        ei      $t6
        luxc1   $f19,$s6($s5)
        madd    $s6,$t5
        madd    $zero,$t1
        maddu   $s3,$gp
        maddu   $t8,$s2
        mfc0    $a2,$14,1
        mfhc1   $s8,$f24
        msub    $s7,$k1
        msubu   $t7,$a1
        mtc0    $t1,$29,3
        mthc1   $zero,$f16
        mul     $s0,$s4,$at
        rdhwr   $sp,$11
        suxc1   $f12,$k1($t5)
