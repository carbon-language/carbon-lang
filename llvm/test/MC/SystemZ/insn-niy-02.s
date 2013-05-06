# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: niy	-524289, 0
#CHECK: error: invalid operand
#CHECK: niy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: niy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: niy	0, -1
#CHECK: error: invalid operand
#CHECK: niy	0, 256

	niy	-524289, 0
	niy	524288, 0
	niy	0(%r1,%r2), 0
	niy	0, -1
	niy	0, 256
