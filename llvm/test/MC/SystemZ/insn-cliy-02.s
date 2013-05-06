# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cliy	-524289, 0
#CHECK: error: invalid operand
#CHECK: cliy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cliy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: cliy	0, -1
#CHECK: error: invalid operand
#CHECK: cliy	0, 256

	cliy	-524289, 0
	cliy	524288, 0
	cliy	0(%r1,%r2), 0
	cliy	0, -1
	cliy	0, 256
