# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: iill	%r0, -1
#CHECK: error: invalid operand
#CHECK: iill	%r0, 0x10000

	iill	%r0, -1
	iill	%r0, 0x10000
