# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lcxbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: lcxbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: lcxbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: lcxbr	%f14, %f0

	lcxbr	%f0, %f2
	lcxbr	%f0, %f14
	lcxbr	%f2, %f0
	lcxbr	%f14, %f0

