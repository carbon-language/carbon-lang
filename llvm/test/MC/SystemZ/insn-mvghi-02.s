# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mvghi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvghi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvghi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvghi	0, -32769
#CHECK: error: invalid operand
#CHECK: mvghi	0, 32768

	mvghi	-1, 0
	mvghi	4096, 0
	mvghi	0(%r1,%r2), 0
	mvghi	0, -32769
	mvghi	0, 32768
