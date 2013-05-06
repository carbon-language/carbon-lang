# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mviy	-524289, 0
#CHECK: error: invalid operand
#CHECK: mviy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mviy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mviy	0, -1
#CHECK: error: invalid operand
#CHECK: mviy	0, 256

	mviy	-524289, 0
	mviy	524288, 0
	mviy	0(%r1,%r2), 0
	mviy	0, -1
	mviy	0, 256
