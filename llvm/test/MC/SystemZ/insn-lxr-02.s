# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lxr	%f2, %f0
#CHECK: error: invalid register
#CHECK: lxr	%f15, %f0
#CHECK: error: invalid register
#CHECK: lxr	%f0, %f2
#CHECK: error: invalid register
#CHECK: lxr	%f0, %f15

	lxr	%f2, %f0
	lxr	%f15, %f0
	lxr	%f0, %f2
	lxr	%f0, %f15
