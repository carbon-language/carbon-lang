# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: c	%r0, -1
#CHECK: error: invalid operand
#CHECK: c	%r0, 4096

	c	%r0, -1
	c	%r0, 4096
