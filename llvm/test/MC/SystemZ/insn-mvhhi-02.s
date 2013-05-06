# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mvhhi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvhhi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvhhi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvhhi	0, -32769
#CHECK: error: invalid operand
#CHECK: mvhhi	0, 32768

	mvhhi	-1, 0
	mvhhi	4096, 0
	mvhhi	0(%r1,%r2), 0
	mvhhi	0, -32769
	mvhhi	0, 32768
