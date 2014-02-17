# RUN: llvm-mc -triple i386 -filetype asm -o - %s | FileCheck %s

	.macro	it, cond
	.endm

	it ne
	.long 1

# CHECK: .long 1

	.macro double first = -1, second = -1
		# begin entry
		.long \first
		.long \second
		# end entry
	.endm

	double
# CHECK: .long -1
# CHECK: .long -1

	double 1
# CHECK: .long 1
# CHECK: .long -1

	double 2, 3
# CHECK: .long 2
# CHECK: .long 3

	double , 4
# CHECK: .long -1
# CHECK: .long 4

	double 5, second = 6
# CHECK: .long 5
# CHECK: .long 6

	double first = 7
# CHECK: .long 7
# CHECK: .long -1

	double second = 8
# CHECK: .long -1
# CHECK: .long 8

	double second = 9, first = 10
# CHECK: .long 10
# CHECK: .long 9

	double second + 11
# CHECK: .long second+11
# CHECK: .long -1

	double , second + 12
# CHECK: .long -1
# CHECK: .long second+12

	double second
# CHECK: .long second
# CHECK: .long -1

	.macro mixed arg0 = 0, arg1 = 1 arg2 = 2, arg3 = 3
		# begin entry
		.long \arg0
		.long \arg1
		.long \arg2
		.long \arg3
		# end entry
	.endm

mixed 1, 2 3

# CHECK: .long 1
# CHECK: .long 2
# CHECK: .long 3
# CHECK: .long 3

mixed 1 2, 3

# CHECK: .long 1
# CHECK: .long 2
# CHECK: .long 3
# CHECK: .long 3

mixed 1 2, 3 4

# CHECK: .long 1
# CHECK: .long 2
# CHECK: .long 3
# CHECK: .long 4

