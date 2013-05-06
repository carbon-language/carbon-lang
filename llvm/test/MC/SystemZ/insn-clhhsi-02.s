# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: clhhsi	-1, 0
#CHECK: error: invalid operand
#CHECK: clhhsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: clhhsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: clhhsi	0, -1
#CHECK: error: invalid operand
#CHECK: clhhsi	0, 65536

	clhhsi	-1, 0
	clhhsi	4096, 0
	clhhsi	0(%r1,%r2), 0
	clhhsi	0, -1
	clhhsi	0, 65536
