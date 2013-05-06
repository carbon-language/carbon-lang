# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mhi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: mhi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: mhi	%r0, foo

	mhi	%r0, -32769
	mhi	%r0, 32768
	mhi	%r0, foo
