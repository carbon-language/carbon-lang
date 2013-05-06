# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: chsi	-1, 0
#CHECK: error: invalid operand
#CHECK: chsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: chsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: chsi	0, -32769
#CHECK: error: invalid operand
#CHECK: chsi	0, 32768

	chsi	-1, 0
	chsi	4096, 0
	chsi	0(%r1,%r2), 0
	chsi	0, -32769
	chsi	0, 32768
