# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: axbr	%f0, %f2
#CHECK: error: invalid register
#CHECK: axbr	%f0, %f14
#CHECK: error: invalid register
#CHECK: axbr	%f2, %f0
#CHECK: error: invalid register
#CHECK: axbr	%f14, %f0

	axbr	%f0, %f2
	axbr	%f0, %f14
	axbr	%f2, %f0
	axbr	%f14, %f0

