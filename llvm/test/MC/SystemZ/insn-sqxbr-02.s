# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: sqxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: sqxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: sqxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: sqxbr	%f14, %f0

	sqxbr	%f0, %f2
	sqxbr	%f0, %f14
	sqxbr	%f2, %f0
	sqxbr	%f14, %f0

