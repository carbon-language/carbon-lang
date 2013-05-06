# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cfdbr	%r0, 0, %r0
#CHECK: error: invalid register
#CHECK: cfdbr	%f0, 0, %f0
#CHECK: error: invalid operand
#CHECK: cfdbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cfdbr	%r0, 16, %f0

	cfdbr	%r0, 0, %r0
	cfdbr	%f0, 0, %f0
	cfdbr	%r0, -1, %f0
	cfdbr	%r0, 16, %f0
