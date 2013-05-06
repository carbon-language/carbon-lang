# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mlg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: mlg	%r0, 524288
#CHECK: error: invalid register
#CHECK: mlg	%r1, 0
#CHECK: error: invalid register
#CHECK: mlg	%r15, 0

	mlg	%r0, -524289
	mlg	%r0, 524288
	mlg	%r1, 0
	mlg	%r15, 0
