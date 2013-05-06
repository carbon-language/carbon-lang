# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: clghsi	-1, 0
#CHECK: error: invalid operand
#CHECK: clghsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: clghsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: clghsi	0, -1
#CHECK: error: invalid operand
#CHECK: clghsi	0, 65536

	clghsi	-1, 0
	clghsi	4096, 0
	clghsi	0(%r1,%r2), 0
	clghsi	0, -1
	clghsi	0, 65536
