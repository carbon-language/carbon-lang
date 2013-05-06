# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: sqdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: sqdb	%f0, 4096

	sqdb	%f0, -1
	sqdb	%f0, 4096
