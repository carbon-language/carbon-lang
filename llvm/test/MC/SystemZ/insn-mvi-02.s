# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mvi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvi	0, -1
#CHECK: error: invalid operand
#CHECK: mvi	0, 256

	mvi	-1, 0
	mvi	4096, 0
	mvi	0(%r1,%r2), 0
	mvi	0, -1
	mvi	0, 256
