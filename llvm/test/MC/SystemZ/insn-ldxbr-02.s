# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: ldxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: ldxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: ldxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: ldxbr	%f14, %f0

	ldxbr	%f0, %f2
	ldxbr	%f0, %f14
	ldxbr	%f2, %f0
	ldxbr	%f14, %f0
