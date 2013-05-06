# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: chhsi	-1, 0
#CHECK: error: invalid operand
#CHECK: chhsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: chhsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: chhsi	0, -32769
#CHECK: error: invalid operand
#CHECK: chhsi	0, 32768

	chhsi	-1, 0
	chhsi	4096, 0
	chhsi	0(%r1,%r2), 0
	chhsi	0, -32769
	chhsi	0, 32768
