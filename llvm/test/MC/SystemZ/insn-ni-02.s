# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: ni	-1, 0
#CHECK: error: invalid operand
#CHECK: ni	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: ni	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: ni	0, -1
#CHECK: error: invalid operand
#CHECK: ni	0, 256

	ni	-1, 0
	ni	4096, 0
	ni	0(%r1,%r2), 0
	ni	0, -1
	ni	0, 256
