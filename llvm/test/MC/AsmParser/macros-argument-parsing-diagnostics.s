# RUN: not llvm-mc -triple i386 -filetype asm -o /dev/null %s 2>&1 | FileCheck %s

	.macro double first = -1, second = -1
		# begin entry
		.long \first
		.long \second
		# end entry
	.endm

	double 0, 1, 2
# CHECK: error: too many positional arguments
# CHECK: 	double 0, 1, 2
# CHECK:                     ^

	double second = 1, 2
# CHECK: error: cannot mix positional and keyword arguments
# CHECK: 	double second = 1, 2
# CHECK:                           ^

	double third = 0
# CHECK: error: parameter named 'third' does not exist for macro 'double'
# CHECK: 	double third = 0
# CHECK:               ^

