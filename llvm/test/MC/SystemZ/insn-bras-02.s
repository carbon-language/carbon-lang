# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: offset out of range
#CHECK: bras	%r0, -0x100002
#CHECK: error: offset out of range
#CHECK: bras	%r0, -1
#CHECK: error: offset out of range
#CHECK: bras	%r0, 1
#CHECK: error: offset out of range
#CHECK: bras	%r0, 0x10000

	bras	%r0, -0x100002
	bras	%r0, -1
	bras	%r0, 1
	bras	%r0, 0x10000
