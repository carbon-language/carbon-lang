# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: fiebr	%r0, 0, %f0
#CHECK: error: invalid register
#CHECK: fiebr	%f0, 0, %r0
#CHECK: error: invalid operand
#CHECK: fiebr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: fiebr	%f0, 16, %f0

	fiebr	%r0, 0, %f0
	fiebr	%f0, 0, %r0
	fiebr	%f0, -1, %f0
	fiebr	%f0, 16, %f0
