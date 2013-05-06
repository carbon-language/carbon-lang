# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: dxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: dxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: dxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: dxbr	%f14, %f0

	dxbr	%f0, %f2
	dxbr	%f0, %f14
	dxbr	%f2, %f0
	dxbr	%f14, %f0

