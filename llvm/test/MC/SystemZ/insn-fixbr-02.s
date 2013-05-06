# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: fixbr	%r0, 0, %f0
#CHECK: error: invalid register
#CHECK: fixbr	%f0, 0, %r0
#CHECK: error: invalid operand
#CHECK: fixbr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: fixbr	%f0, 16, %f0
#CHECK: error: invalid register
#CHECK: fixbr	%f0, 0, %f2
#CHECK: error: invalid register
#CHECK: fixbr	%f0, 0, %f14
#CHECK: error: invalid register
#CHECK: fixbr	%f2, 0, %f0
#CHECK: error: invalid register
#CHECK: fixbr	%f14, 0, %f0

	fixbr	%r0, 0, %f0
	fixbr	%f0, 0, %r0
	fixbr	%f0, -1, %f0
	fixbr	%f0, 16, %f0
	fixbr	%f0, 0, %f2
	fixbr	%f0, 0, %f14
	fixbr	%f2, 0, %f0
	fixbr	%f14, 0, %f0
