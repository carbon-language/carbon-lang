# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        bc1tl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc1tl $fcc1,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc1fl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc1fl $fcc1,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc2tl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc2tl $fcc1,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc2fl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc2fl $fcc1,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
