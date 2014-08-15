# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips4 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        madd.d          $f18,$f19,$f26,$f20
        madd.s          $f1,$f31,$f19,$f25
        msub.d          $f10,$f1,$f31,$f18
        msub.s          $f12,$f19,$f10,$f16
        nmadd.d         $f18,$f9,$f14,$f19
        nmadd.s         $f0,$f5,$f25,$f12
        nmsub.d         $f30,$f8,$f16,$f30
        nmsub.s         $f1,$f24,$f19,$f4
        recip.d         $f19,$f6
        recip.s         $f3,$f30
        rsqrt.d         $f3,$f28
        rsqrt.s         $f4,$f8
