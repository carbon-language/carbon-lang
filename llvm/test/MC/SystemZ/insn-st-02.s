# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: st	%r0, -1
#CHECK: error: invalid operand
#CHECK: st	%r0, 4096

	st	%r0, -1
	st	%r0, 4096
