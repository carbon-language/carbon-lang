# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cl	%r0, -1
#CHECK: error: invalid operand
#CHECK: cl	%r0, 4096

	cl	%r0, -1
	cl	%r0, 4096
