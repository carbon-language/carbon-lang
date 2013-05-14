# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, 0x100000000

	clgfrl	%r0, -0x1000000002
	clgfrl	%r0, -1
	clgfrl	%r0, 1
	clgfrl	%r0, 0x100000000
