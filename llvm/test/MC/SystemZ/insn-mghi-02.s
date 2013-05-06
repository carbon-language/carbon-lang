# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: mghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: mghi	%r0, foo

	mghi	%r0, -32769
	mghi	%r0, 32768
	mghi	%r0, foo
