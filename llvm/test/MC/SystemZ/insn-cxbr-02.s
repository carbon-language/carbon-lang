# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: cxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: cxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: cxbr	%f14, %f0

	cxbr	%f0, %f2
	cxbr	%f0, %f14
	cxbr	%f2, %f0
	cxbr	%f14, %f0

