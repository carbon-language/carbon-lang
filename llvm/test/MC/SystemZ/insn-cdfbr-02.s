# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cdfbr	%r0, %r0
#CHECK: error: invalid register
#CHECK: cdfbr	%f0, %f0
#CHECK: error: invalid register
#CHECK: cdfbr	%f0, %a0
#CHECK: error: invalid register
#CHECK: cdfbr	%a0, %r0

	cdfbr	%r0, %r0
	cdfbr	%f0, %f0
	cdfbr	%f0, %a0
	cdfbr	%a0, %r0
