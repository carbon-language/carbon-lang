# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: mxdb	%f2, 0
#CHECK: error: invalid register
#CHECK: mxdb	%f15, 0
#CHECK: error: invalid operand
#CHECK: mxdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: mxdb	%f0, 4096

	mxdb	%f2, 0
	mxdb	%f15, 0
	mxdb	%f0, -1
	mxdb	%f0, 4096
