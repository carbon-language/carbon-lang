# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mh	%r0, -1
#CHECK: error: invalid operand
#CHECK: mh	%r0, 4096

	mh	%r0, -1
	mh	%r0, 4096
