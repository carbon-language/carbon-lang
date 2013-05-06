# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: aghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: aghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: aghi	%r0, foo

	aghi	%r0, -32769
	aghi	%r0, 32768
	aghi	%r0, foo
