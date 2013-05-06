# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cxgbr	%r0, %r0
#CHECK: error: invalid register
#CHECK: cxgbr	%f0, %f0
#CHECK: error: invalid register
#CHECK: cxgbr	%f0, %a0
#CHECK: error: invalid register
#CHECK: cxgbr	%a0, %r0
#CHECK: error: invalid register
#CHECK: cxgbr	%f2, %r0
#CHECK: error: invalid register
#CHECK: cxgbr	%f14, %r0

	cxgbr	%r0, %r0
	cxgbr	%f0, %f0
	cxgbr	%f0, %a0
	cxgbr	%a0, %r0
	cxgbr	%f2, %r0
	cxgbr	%f14, %r0
