# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: xi	-1, 0
#CHECK: error: invalid operand
#CHECK: xi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: xi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: xi	0, -1
#CHECK: error: invalid operand
#CHECK: xi	0, 256

	xi	-1, 0
	xi	4096, 0
	xi	0(%r1,%r2), 0
	xi	0, -1
	xi	0, 256
