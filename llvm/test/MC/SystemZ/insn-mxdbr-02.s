# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: mxdbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: mxdbr	%f15, %f0

	mxdbr	%f2, %f0
	mxdbr	%f15, %f0
