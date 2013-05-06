# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: s	%r0, -1
#CHECK: error: invalid operand
#CHECK: s	%r0, 4096

	s	%r0, -1
	s	%r0, 4096
