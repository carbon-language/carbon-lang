# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        pause # CHECK: requires a CPU feature not currently enabled
