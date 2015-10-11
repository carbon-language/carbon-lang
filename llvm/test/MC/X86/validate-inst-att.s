# RUN: not llvm-mc -triple i686 -filetype asm -o /dev/null %s 2>&1 | FileCheck %s

	.text
	int $65535
# CHECK: error: invalid operand for instruction
# CHECK:	int $65535
# CHECK:            ^

	int $-129
# CHECK: error: invalid operand for instruction
# CHECK:	int $-129
# CHECK:            ^
