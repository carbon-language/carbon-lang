# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: chi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: chi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: chi	%r0, foo

	chi	%r0, -32769
	chi	%r0, 32768
	chi	%r0, foo
