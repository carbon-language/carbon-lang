# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cli	-1, 0
#CHECK: error: invalid operand
#CHECK: cli	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cli	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: cli	0, -1
#CHECK: error: invalid operand
#CHECK: cli	0, 256

	cli	-1, 0
	cli	4096, 0
	cli	0(%r1,%r2), 0
	cli	0, -1
	cli	0, 256
