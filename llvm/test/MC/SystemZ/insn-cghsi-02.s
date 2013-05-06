# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cghsi	-1, 0
#CHECK: error: invalid operand
#CHECK: cghsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cghsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: cghsi	0, -32769
#CHECK: error: invalid operand
#CHECK: cghsi	0, 32768

	cghsi	-1, 0
	cghsi	4096, 0
	cghsi	0(%r1,%r2), 0
	cghsi	0, -32769
	cghsi	0, 32768
