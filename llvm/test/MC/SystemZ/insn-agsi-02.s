# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: agsi	-524289, 0
#CHECK: error: invalid operand
#CHECK: agsi	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: agsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: agsi	0, -129
#CHECK: error: invalid operand
#CHECK: agsi	0, 128

	agsi	-524289, 0
	agsi	524288, 0
	agsi	0(%r1,%r2), 0
	agsi	0, -129
	agsi	0, 128
