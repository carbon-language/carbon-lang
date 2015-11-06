# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 -mattr=+msa \
# RUN:     -show-encoding 2>%t1
# RUN: FileCheck %s < %t1

    .set noat
    insve.b $w25[3], $w9[1] # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.h $w24[2], $w2[1] # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.w $w0[2], $w13[1] # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.d $w3[0], $w18[1] # CHECK: :[[@LINE]]:26: error: expected '0'
