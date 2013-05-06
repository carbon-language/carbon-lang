# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cfebr	%r0, 0, %r0
#CHECK: error: invalid register
#CHECK: cfebr	%f0, 0, %f0
#CHECK: error: invalid operand
#CHECK: cfebr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cfebr	%r0, 16, %f0

	cfebr	%r0, 0, %r0
	cfebr	%f0, 0, %f0
	cfebr	%r0, -1, %f0
	cfebr	%r0, 16, %f0
