# Instructions that are available for the current ISA but should be rejected by
# the assembler (e.g. invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips64-unknown-linux -mcpu=mips64r6 2>%t1
# RUN: FileCheck %s < %t1 -check-prefix=ASM

        .text
        .set noreorder
	.set noat
        jalr.hb $31 # ASM: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31 # ASM: :[[@LINE]]:9: error: source and destination must be different
        ldc2    $8,-21181($at)   # ASM: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        break 1024        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        break 1024, 5     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        break 7, 1024     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        break 1024, 1024  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
