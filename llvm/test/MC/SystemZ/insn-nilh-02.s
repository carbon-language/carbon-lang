# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: nilh	%r0, -1
#CHECK: error: invalid operand
#CHECK: nilh	%r0, 0x10000

	nilh	%r0, -1
	nilh	%r0, 0x10000
