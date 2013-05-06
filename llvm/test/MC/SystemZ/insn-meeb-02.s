# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: meeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: meeb	%f0, 4096

	meeb	%f0, -1
	meeb	%f0, 4096
