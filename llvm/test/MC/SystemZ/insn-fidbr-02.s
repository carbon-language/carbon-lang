# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: fidbr	%r0, 0, %f0
#CHECK: error: invalid register
#CHECK: fidbr	%f0, 0, %r0
#CHECK: error: invalid operand
#CHECK: fidbr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: fidbr	%f0, 16, %f0

	fidbr	%r0, 0, %f0
	fidbr	%f0, 0, %r0
	fidbr	%f0, -1, %f0
	fidbr	%f0, 16, %f0
