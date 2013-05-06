# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: la	%r0, -1
#CHECK: error: invalid operand
#CHECK: la	%r0, 4096

	la	%r0, -1
	la	%r0, 4096
