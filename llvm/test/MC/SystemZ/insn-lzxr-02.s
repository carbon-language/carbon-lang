# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lzxr	%f2
#CHECK: error: invalid register
#CHECK: lzxr	%f14
#CHECK: error: invalid register
#CHECK: lzxr	%f15

	lzxr	%f2
	lzxr	%f14
	lzxr	%f15
