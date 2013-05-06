# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: nihh	%r0, -1
#CHECK: error: invalid operand
#CHECK: nihh	%r0, 0x10000

	nihh	%r0, -1
	nihh	%r0, 0x10000
