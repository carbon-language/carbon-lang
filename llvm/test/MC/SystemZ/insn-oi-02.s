# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: oi	-1, 0
#CHECK: error: invalid operand
#CHECK: oi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: oi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: oi	0, -1
#CHECK: error: invalid operand
#CHECK: oi	0, 256

	oi	-1, 0
	oi	4096, 0
	oi	0(%r1,%r2), 0
	oi	0, -1
	oi	0, 256
