# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: ste	%f0, -1
#CHECK: error: invalid operand
#CHECK: ste	%f0, 4096

	ste	%f0, -1
	ste	%f0, 4096
