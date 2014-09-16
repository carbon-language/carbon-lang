# Instructions that are invalid but currently emit the wrong error message.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        bc1any2f  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc1any2t  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc1any4f  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc1any4t  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
