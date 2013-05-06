# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lnxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: lnxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: lnxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: lnxbr	%f14, %f0

	lnxbr	%f0, %f2
	lnxbr	%f0, %f14
	lnxbr	%f2, %f0
	lnxbr	%f14, %f0

