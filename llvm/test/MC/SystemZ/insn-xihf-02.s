# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: xihf	%r0, -1
#CHECK: error: invalid operand
#CHECK: xihf	%r0, 1 << 32

	xihf	%r0, -1
	xihf	%r0, 1 << 32
