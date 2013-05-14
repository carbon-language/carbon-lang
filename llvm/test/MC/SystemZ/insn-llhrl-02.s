# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: offset out of range
#CHECK: llhrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: llhrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: llhrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: llhrl	%r0, 0x100000000

	llhrl	%r0, -0x1000000002
	llhrl	%r0, -1
	llhrl	%r0, 1
	llhrl	%r0, 0x100000000
