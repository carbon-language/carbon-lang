# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: ahi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: ahi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: ahi	%r0, foo

	ahi	%r0, -32769
	ahi	%r0, 32768
	ahi	%r0, foo
