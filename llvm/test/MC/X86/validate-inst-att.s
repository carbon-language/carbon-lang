# RUN: not llvm-mc -triple i686 -filetype asm -o /dev/null %s 2>&1 | FileCheck %s

	.text
	int $65535
# CHECK: error: interrupt vector must be in range [0-255]
# CHECK:	int $65535
# CHECK:            ^
