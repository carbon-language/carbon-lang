# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: xiy	-524289, 0
#CHECK: error: invalid operand
#CHECK: xiy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: xiy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: xiy	0, -1
#CHECK: error: invalid operand
#CHECK: xiy	0, 256

	xiy	-524289, 0
	xiy	524288, 0
	xiy	0(%r1,%r2), 0
	xiy	0, -1
	xiy	0, 256
