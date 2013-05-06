# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lpxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: lpxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: lpxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: lpxbr	%f14, %f0

	lpxbr	%f0, %f2
	lpxbr	%f0, %f14
	lpxbr	%f2, %f0
	lpxbr	%f14, %f0

