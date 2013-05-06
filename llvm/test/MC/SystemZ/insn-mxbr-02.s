# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: mxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: mxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: mxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: mxbr	%f14, %f0

	mxbr	%f0, %f2
	mxbr	%f0, %f14
	mxbr	%f2, %f0
	mxbr	%f14, %f0

