# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: asi	-524289, 0
#CHECK: error: invalid operand
#CHECK: asi	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: asi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: asi	0, -129
#CHECK: error: invalid operand
#CHECK: asi	0, 128

	asi	-524289, 0
	asi	524288, 0
	asi	0(%r1,%r2), 0
	asi	0, -129
	asi	0, 128
