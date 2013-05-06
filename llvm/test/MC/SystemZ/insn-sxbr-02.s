# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: sxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: sxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: sxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: sxbr	%f14, %f0

	sxbr	%f0, %f2
	sxbr	%f0, %f14
	sxbr	%f2, %f0
	sxbr	%f14, %f0

