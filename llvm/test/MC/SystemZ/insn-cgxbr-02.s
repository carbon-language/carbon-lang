# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cgxbr	%r0, 0, %r0
#CHECK: error: invalid register
#CHECK: cgxbr	%f0, 0, %f0
#CHECK: error: invalid operand
#CHECK: cgxbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgxbr	%r0, 16, %f0
#CHECK: error: invalid register
#CHECK: cgxbr	%r0, 0, %f2
#CHECK: error: invalid register
#CHECK: cgxbr	%r0, 0, %f14

	cgxbr	%r0, 0, %r0
	cgxbr	%f0, 0, %f0
	cgxbr	%r0, -1, %f0
	cgxbr	%r0, 16, %f0
	cgxbr	%r0, 0, %f2
	cgxbr	%r0, 0, %f14

