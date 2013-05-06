# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: nihl	%r0, -1
#CHECK: error: invalid operand
#CHECK: nihl	%r0, 0x10000

	nihl	%r0, -1
	nihl	%r0, 0x10000
