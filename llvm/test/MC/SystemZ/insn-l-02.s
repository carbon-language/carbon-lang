# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: l	%r0, -1
#CHECK: error: invalid operand
#CHECK: l	%r0, 4096

	l	%r0, -1
	l	%r0, 4096
