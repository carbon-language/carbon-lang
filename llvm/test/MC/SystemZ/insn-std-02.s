# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: std	%f0, -1
#CHECK: error: invalid operand
#CHECK: std	%f0, 4096

	std	%f0, -1
	std	%f0, 4096
