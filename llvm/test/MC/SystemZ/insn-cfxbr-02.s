# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cfxbr	%r0, 0, %r0
#CHECK: error: invalid register
#CHECK: cfxbr	%f0, 0, %f0
#CHECK: error: invalid operand
#CHECK: cfxbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cfxbr	%r0, 16, %f0
#CHECK: error: invalid register
#CHECK: cfxbr	%r0, 0, %f2
#CHECK: error: invalid register
#CHECK: cfxbr	%r0, 0, %f14

	cfxbr	%r0, 0, %r0
	cfxbr	%f0, 0, %f0
	cfxbr	%r0, -1, %f0
	cfxbr	%r0, 16, %f0
	cfxbr	%r0, 0, %f2
	cfxbr	%r0, 0, %f14

