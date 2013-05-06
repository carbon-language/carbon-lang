# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cxfbr	%r0, %r0
#CHECK: error: invalid register
#CHECK: cxfbr	%f0, %f0
#CHECK: error: invalid register
#CHECK: cxfbr	%f0, %a0
#CHECK: error: invalid register
#CHECK: cxfbr	%a0, %r0
#CHECK: error: invalid register
#CHECK: cxfbr	%f2, %r0
#CHECK: error: invalid register
#CHECK: cxfbr	%f14, %r0

	cxfbr	%r0, %r0
	cxfbr	%f0, %f0
	cxfbr	%f0, %a0
	cxfbr	%a0, %r0
	cxfbr	%f2, %r0
	cxfbr	%f14, %r0
