# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mvhi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvhi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvhi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvhi	0, -32769
#CHECK: error: invalid operand
#CHECK: mvhi	0, 32768

	mvhi	-1, 0
	mvhi	4096, 0
	mvhi	0(%r1,%r2), 0
	mvhi	0, -32769
	mvhi	0, 32768
