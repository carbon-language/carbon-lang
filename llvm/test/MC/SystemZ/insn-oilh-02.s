# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: oilh	%r0, -1
#CHECK: error: invalid operand
#CHECK: oilh	%r0, 0x10000

	oilh	%r0, -1
	oilh	%r0, 0x10000
