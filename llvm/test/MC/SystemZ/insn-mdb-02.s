# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: mdb	%f0, 4096

	mdb	%f0, -1
	mdb	%f0, 4096
