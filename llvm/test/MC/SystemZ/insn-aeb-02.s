# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: aeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: aeb	%f0, 4096

	aeb	%f0, -1
	aeb	%f0, 4096
