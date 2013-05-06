# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: sqeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: sqeb	%f0, 4096

	sqeb	%f0, -1
	sqeb	%f0, 4096
