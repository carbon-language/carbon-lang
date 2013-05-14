# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: offset out of range
#CHECK: chrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: chrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: chrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: chrl	%r0, 0x100000000

	chrl	%r0, -0x1000000002
	chrl	%r0, -1
	chrl	%r0, 1
	chrl	%r0, 0x100000000
