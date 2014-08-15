# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        rorv            $13,$a3,$s5
