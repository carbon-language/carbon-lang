# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        beql $1,$2,4            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bgezall $3,8            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bgezl $3,8              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bgtzl $4,16             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        blezl $3,8              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bltzall $3,8            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bltzl $4,16             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bnel $1,$2,4            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc1tl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc1fl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc2tl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        bc2fl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
        prefx 0,$2($31)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: Unknown instruction
